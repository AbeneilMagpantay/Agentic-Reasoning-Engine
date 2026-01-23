
import time
import json
import random
import statistics
import concurrent.futures
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph.nodes.local_grader import LocalHallucinationGrader, GradeRequest

def load_dataset():
    with open("data/golden_dataset.json", "r") as f:
        data = json.load(f)
    return data

def run_stress_test(num_requests=1000, concurrency=10):
    print(f"--- Starting Stress Test: {num_requests} requests, {concurrency} threads ---")
    
    grader = LocalHallucinationGrader()
    dataset = load_dataset()
    
    latencies = []
    errors = 0
    
    start_time = time.time()
    
    def task():
        item = random.choice(dataset)
        text = item['text']
        context_part = text.split("Answer:")[0].replace("Context:", "").strip()
        answer_part = text.split("Answer:")[1].strip()
        req = GradeRequest(context=context_part, answer=answer_part)
        
        try:
            res = grader.grade_sync(req)
            if res:
                return res.latency_ms
            else:
                return -1 # Fallback/Low confidence
        except Exception:
            return -2 # Error

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(task) for _ in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            lat = future.result()
            if lat >= 0:
                latencies.append(lat)
            else:
                errors += 1
                
    total_time = time.time() - start_time
    throughput = num_requests / total_time
    
    print("\n--- Stress Test Results ---")
    print(f"Total Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Errors/Fallbacks: {errors}")
    
    if latencies:
        print(f"Avg Latency: {statistics.mean(latencies):.2f}ms")
        print(f"P50 Latency: {statistics.median(latencies):.2f}ms")
        print(f"P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
        print(f"P99 Latency: {statistics.quantiles(latencies, n=100)[98]:.2f}ms")
        print(f"Min Latency: {min(latencies):.2f}ms")
        print(f"Max Latency: {max(latencies):.2f}ms")
    else:
        print("No successful requests recorded.")

if __name__ == "__main__":
    # Warn user about CPU/GPU usage
    print("WARNING: This test will stress your system.")
    time.sleep(2)
    run_stress_test(num_requests=500, concurrency=4)
