
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

import src.graph.nodes.grader as grader_module

class TestFallback(unittest.TestCase):
    
    @patch('src.graph.nodes.grader.USE_LOCAL_GRADER', True)
    @patch('src.graph.nodes.grader._get_local_grader')
    @patch('src.graph.nodes.grader._grade_with_api')
    def test_low_confidence_fallback(self, mock_api, mock_get_local):
        print("\n--- Testing Fallback: Low Confidence ---")
        # Setup local grader to return low confidence
        mock_local_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.confidence = 0.5 # Below 0.8 threshold
        mock_result.is_faithful = True
        mock_result.latency_ms = 10.0
        
        mock_local_instance.grade_sync.return_value = mock_result
        mock_get_local.return_value = mock_local_instance
        
        # Setup API to return "yes"
        mock_api.return_value = "yes"
        
        state = {"documents": ["doc1"], "question": "q", "route": "rag"}
        
        # Run
        result = grader_module.grade_documents(state)
        
        # Verify
        mock_get_local.assert_called()
        mock_api.assert_called_with("q", "doc1", unittest.mock.ANY)
        print("✅ Low confidence triggered API fallback correctly")
        self.assertEqual(len(result["documents"]), 1)

    @patch('src.graph.nodes.grader.USE_LOCAL_GRADER', True)
    @patch('src.graph.nodes.grader._get_local_grader')
    @patch('src.graph.nodes.grader._grade_with_api')
    def test_error_fallback(self, mock_api, mock_get_local):
        print("\n--- Testing Fallback: Exception ---")
         # Setup local grader to raise exception
        mock_local_instance = MagicMock()
        mock_local_instance.grade_sync.side_effect = RuntimeError("GPU Explosion")
        mock_get_local.return_value = mock_local_instance
        
        mock_api.return_value = "yes"
        
        state = {"documents": ["doc1"], "question": "q", "route": "rag"}
        result = grader_module.grade_documents(state)
        
        mock_api.assert_called()
        print("✅ Runtime error triggered API fallback correctly")
        self.assertEqual(len(result["documents"]), 1)
        
    @patch('src.graph.nodes.grader.USE_LOCAL_GRADER', True)
    @patch('src.graph.nodes.grader._get_local_grader')
    @patch('src.graph.nodes.grader._grade_with_api')
    def test_success_path(self, mock_api, mock_get_local):
        print("\n--- Testing Success: High Confidence ---")
        mock_local_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.confidence = 0.95 # High confidence
        mock_result.is_faithful = True
        mock_result.latency_ms = 10.0
        
        mock_local_instance.grade_sync.return_value = mock_result
        mock_get_local.return_value = mock_local_instance
        
        state = {"documents": ["doc1"], "question": "q", "route": "rag"}
        result = grader_module.grade_documents(state)
        
        mock_get_local.assert_called()
        mock_api.assert_not_called()
        print("✅ High confidence skipped API call correctly")
        self.assertEqual(len(result["documents"]), 1)

if __name__ == "__main__":
    unittest.main()
