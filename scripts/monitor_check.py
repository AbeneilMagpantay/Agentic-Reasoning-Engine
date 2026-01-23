"""
monitor_check.py - Performance Audit Tool for RAG Pipelines

Comprehensive telemetry system for tracking:
- Weighted health scoring (0-100)
- P95 confidence/latency tracking
- Error type attribution
- Executive summary reports
"""

import os
import re
import statistics
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Log file location
LOG_FILE = os.getenv("GRADER_LOG_PATH", "logs/grader.log")

# Token patterns for log parsing
SUCCESS_TOKEN = "SUCCESS: USING LOCAL 10MS GUARDRAIL"
FALLBACK_TOKEN = "FALLBACK: LOCAL UNSURE OR DISABLED"
ERROR_TOKEN = "ERROR"
WARN_TOKEN = "WARNING"

# Regex patterns for numerical extraction
CONF_PATTERN = re.compile(r"[Cc]onfidence[:\s]*(?P<score>0\.\d+)")
LATENCY_PATTERN = re.compile(r"[Ll]atency[:\s]*(?P<ms>\d+\.?\d*)ms")
FALLBACK_REASON_PATTERN = re.compile(r"falling back to API|FALLBACK[:\s]*(.*)")
LOCAL_LOW_CONF_PATTERN = re.compile(r"Local grader low confidence \(([\d.]+)\)")
LOCAL_ERROR_PATTERN = re.compile(r"Local grader error: (.*)")


class EngineAuditor:
    """
    Full health auditor for the Hybrid Grader system.
    """
    
    def __init__(self, log_path: str = LOG_FILE):
        self.log_path = log_path
        self.stats = {
            "total_attempts": 0,
            "local_success": 0,
            "api_fallback": 0,
            "errors": 0,
            "warnings": 0,
            "confidences": [],
            "latencies": [],
            "error_types": Counter(),
            "fallback_reasons": Counter(),
        }
    
    def parse_logs(self) -> bool:
        """Scans the engine logs and populates performance metrics."""
        if not os.path.exists(self.log_path):
            print(f"Critical Error: Log file not found at {self.log_path}")
            return False
        
        with open(self.log_path, "r") as f:
            for line in f:
                self._parse_line(line)
        
        return True
    
    def _parse_line(self, line: str) -> None:
        """Parse a single log line and update stats."""
        
        # Track Basic Throughput
        if SUCCESS_TOKEN in line or "---GRADE: DOCUMENT" in line.upper():
            if "LOCAL" in line.upper() or SUCCESS_TOKEN in line:
                self.stats["local_success"] += 1
                self.stats["total_attempts"] += 1
        
        if FALLBACK_TOKEN in line or "falling back to API" in line.lower():
            self.stats["api_fallback"] += 1
            self.stats["total_attempts"] += 1
            
            # Capture fallback reason
            if match := FALLBACK_REASON_PATTERN.search(line):
                reason = match.group(1) if match.group(1) else "Unknown"
                self.stats["fallback_reasons"][reason[:50]] += 1
        
        # Track Resilience Issues
        if ERROR_TOKEN in line:
            self.stats["errors"] += 1
            # Capture error type
            if match := LOCAL_ERROR_PATTERN.search(line):
                self.stats["error_types"][match.group(1)[:50]] += 1
        
        if WARN_TOKEN in line:
            self.stats["warnings"] += 1
        
        # Extract Numerical Metrics
        if match := CONF_PATTERN.search(line):
            self.stats["confidences"].append(float(match.group("score")))
        
        if match := LATENCY_PATTERN.search(line):
            self.stats["latencies"].append(float(match.group("ms")))
        
        # Low confidence fallback tracking
        if match := LOCAL_LOW_CONF_PATTERN.search(line):
            self.stats["confidences"].append(float(match.group(1)))
    
    def calculate_health_score(self) -> float:
        """
        Generates a weighted health score (0-100).
        
        Penalties:
        - Errors: -15 each
        - API Fallback rate: -30% weight
        - Warnings: -2 each
        """
        if self.stats["total_attempts"] == 0:
            return 0
        
        # Base score is 100
        score = 100
        
        # Error penalty: -15 per error (max -45)
        error_penalty = min(45, self.stats["errors"] * 15)
        
        # Fallback penalty: proportional to fallback rate (max -30)
        fallback_rate = self.stats["api_fallback"] / self.stats["total_attempts"]
        fallback_penalty = fallback_rate * 30
        
        # Warning penalty: -2 per warning (max -10)
        warning_penalty = min(10, self.stats["warnings"] * 2)
        
        score = score - error_penalty - fallback_penalty - warning_penalty
        return max(0, min(100, score))
    
    def run_report(self) -> Dict:
        """Outputs the final health and performance audit."""
        if not self.parse_logs():
            return {"status": "error", "message": "Log file not found"}
        
        total = self.stats["total_attempts"]
        success_rate = (self.stats["local_success"] / total * 100) if total > 0 else 0
        avg_conf = statistics.mean(self.stats["confidences"]) if self.stats["confidences"] else 0
        health = self.calculate_health_score()
        
        # Executive summary
        status = "EXCELLENT" if health > 85 else "STABLE" if health > 65 else "DEGRADED"
        
        print("\n" + "=" * 60)
        print(f" AGENTIC REASONING ENGINE - PHASE 5 AUDIT REPORT")
        print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        print(f"\nSYSTEM HEALTH: {status} ({health:.1f}/100)")
        print(f"Overall Throughput: {total} grading events processed")
        print("-" * 60)
        
        # Efficiency Metrics
        print(f"\n📊 EFFICIENCY METRICS")
        print(f"  Local Guardrail Success:  {self.stats['local_success']} ({success_rate:.1f}%)")
        print(f"  API Fallback Rate:        {self.stats['api_fallback']} ({(100 - success_rate):.1f}%)")
        
        if self.stats["latencies"]:
            avg_lat = statistics.mean(self.stats["latencies"])
            p95_lat = statistics.quantiles(self.stats["latencies"], n=20)[18] if len(self.stats["latencies"]) > 1 else avg_lat
            print(f"  Avg Local Latency:        {avg_lat:.2f}ms (Target: <10ms)")
            print(f"  P95 Latency:              {p95_lat:.2f}ms")
        
        print("-" * 60)
        
        # Model Quality
        print(f"\n🎯 MODEL QUALITY")
        print(f"  Avg Confidence:           {avg_conf:.4f} (Threshold: 0.8)")
        if self.stats["confidences"]:
            p95_conf = statistics.quantiles(self.stats["confidences"], n=20)[18] if len(self.stats["confidences"]) > 1 else avg_conf
            print(f"  P95 Confidence:           {p95_conf:.4f}")
        
        # Error Audit
        if self.stats["errors"] > 0:
            print("-" * 60)
            print(f"\n⚠️  CRITICAL ERRORS: {self.stats['errors']}")
            for msg, count in self.stats["error_types"].most_common(3):
                print(f"    > {count}x: {msg}...")
        
        # Fallback Reasons
        if self.stats["fallback_reasons"]:
            print("-" * 60)
            print(f"\n📉 FALLBACK REASONS:")
            for reason, count in self.stats["fallback_reasons"].most_common(3):
                print(f"    > {count}x: {reason}")
        
        print("\n" + "=" * 60)
        print(f"Target: Local Rate >80% | Latency <15ms | Confidence >0.8")
        print("=" * 60 + "\n")
        
        return {
            "status": status.lower(),
            "health_score": health,
            "local_success_rate": success_rate,
            "api_fallback_rate": 100 - success_rate,
            "avg_confidence": avg_conf,
            "avg_latency_ms": statistics.mean(self.stats["latencies"]) if self.stats["latencies"] else 0,
            "total_requests": total,
            "errors": self.stats["errors"],
        }


def check_health(log_path: str = LOG_FILE) -> Dict:
    """Quick health check for API integration."""
    auditor = EngineAuditor(log_path)
    return auditor.run_report()


if __name__ == "__main__":
    import sys
    
    log_path = sys.argv[1] if len(sys.argv) > 1 else LOG_FILE
    
    auditor = EngineAuditor(log_path)
    auditor.run_report()
