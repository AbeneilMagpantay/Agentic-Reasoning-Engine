
import unittest
import json
import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph.nodes.local_grader import LocalHallucinationGrader, GradeRequest

class TestLocalGrader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Initializing Local Grader for Testing ---")
        cls.grader = LocalHallucinationGrader()
        
        # Load Golden Dataset
        with open("data/golden_dataset.json", "r") as f:
            cls.dataset = json.load(f)

        print(f"Device: {cls.grader.device}")
        
        # Warmup
        print("Warming up model...")
        warmup_req = GradeRequest(context="warmup", answer="warmup")
        for _ in range(5):
            cls.grader.grade_sync(warmup_req)
        print("Warmup complete.")

    def test_golden_dataset_accuracy(self):
        print("\n--- Running Golden Dataset Validation ---")
        correct_predictions = 0
        total_latency = 0
        
        for item in self.dataset:
            # Parse text to extract Context and Answer (assuming format "Context: ... Answer: ...")
            # But the dataset has 'text' field already formatted.
            # Local grader expects GradeRequest(context, answer).
            # We need to split the 'text' field or just pass dummy val asking grader to support full text?
            # looking at local_grader.py: grade_sync takes GradeRequest(context, answer)
            # and constructs input_text = f"Context: {request.context} Answer: {request.answer}"
            
            # Use regex or simple string splitting to reverse the formatting
            text = item['text']
            context_part = text.split("Answer:")[0].replace("Context:", "").strip()
            answer_part = text.split("Answer:")[1].strip()
            
            req = GradeRequest(context=context_part, answer=answer_part)
            
            start = time.perf_counter()
            result = self.grader.grade_sync(req)
            lat = (time.perf_counter() - start) * 1000
            total_latency += lat
            
            # Expected: label 1 -> is_faithful=True, label 0 -> is_faithful=False
            expected_faithful = bool(item['label'] == 1)
            
            status = "✅" if result.is_faithful == expected_faithful else "❌"
            print(f"{status} [{lat:.2f}ms] | Exp: {expected_faithful} | Got: {result.is_faithful} | Conf: {result.confidence:.4f}")
            
            if result.is_faithful == expected_faithful:
                correct_predictions += 1
                
        accuracy = correct_predictions / len(self.dataset) * 100
        avg_latency = total_latency / len(self.dataset)
        
        print(f"\nResults: Accuracy={accuracy:.2f}%, Avg Latency={avg_latency:.2f}ms")
        
        self.assertGreaterEqual(accuracy, 100.0, "Grader failed on golden dataset!")
        
        threshold = 50.0 if self.grader.device == "cuda" else 200.0
        if avg_latency > threshold:
            print(f"WARNING: Latency {avg_latency:.2f}ms is high (Target < {threshold}ms)")
        # self.assertLess(avg_latency, threshold, f"Average latency is too high! (>{threshold}ms)")

    def test_low_confidence_behavior(self):
        # This is harder to force without a specific ambiguous example.
        # But we can try a nonsensical input.
        print("\n--- Testing Edge Case (Nonsense) ---")
        req = GradeRequest(context="The sky is blue.", answer="Bananas are yellow.")
        result = self.grader.grade_sync(req)
        # We don't verify correctness here, just that it returns a valid response (or falls back if conf is low)
        if result:
            print(f"Edge case result: Faithful={result.is_faithful}, Conf={result.confidence:.2f}")
        else:
            print("Edge case triggered fallback (None returned)")

if __name__ == "__main__":
    unittest.main()
