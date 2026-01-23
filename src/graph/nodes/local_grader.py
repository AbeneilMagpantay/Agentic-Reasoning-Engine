"""
local_grader.py - Local ModernBERT Hallucination Grader

Replaces cloud API calls with low-latency local inference.
Supports hot-swappable fallback logic.
"""

import torch
from typing import Optional
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class GradeRequest(BaseModel):
    """Input schema for grading requests."""
    context: str
    answer: str


class GradeResponse(BaseModel):
    """Output schema for grading results."""
    is_faithful: bool
    confidence: float
    latency_ms: float


class ModernBERTClassifier(nn.Module):
    """ModernBERT with classification head - matches training notebook."""
    
    def __init__(self, model_name: str = 'answerdotai/ModernBERT-base', num_labels: int = 2, quantization_config=None, attn_implementation=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
            dtype=torch.float16 if quantization_config or attn_implementation else torch.float32
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class LocalHallucinationGrader:
    """
    Local 10ms Guardrail using fine-tuned ModernBERT.
    
    Optimized for production use with hot-swapping capabilities.
    """
    
    def __init__(
        self,
        model_path: str = "./models/guardrail_v1.pt",
        base_model: str = "answerdotai/ModernBERT-base",
        use_quantization: bool = True,
        use_flash_attn: bool = True
    ):
        """
        Initialize the local 10ms Guardrail.
        
        Args:
            model_path: Path to fine-tuned weights (.pt file)
            base_model: Base model for tokenizer
            use_quantization: Use 4-bit NF4 quantization (requires bitsandbytes)
            use_flash_attn: Use Flash Attention 2 for 2x speedup
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LocalGrader] Initializing on {self.device}")
        
        # 1. Configure Quantization (NF4)
        quant_config = None
        if use_quantization and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            print("[LocalGrader] NF4 4-bit Quantization Enabled")

        # 2. Configure Flash Attention
        attn_impl = None
        if use_flash_attn and self.device == "cuda":
            attn_impl = "flash_attention_2"
            print("[LocalGrader] Flash Attention 2 Enabled")
        
        # 3. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # 4. Load model architecture with optimizations
        self.model = ModernBERTClassifier(
            base_model, 
            quantization_config=quant_config,
            attn_implementation=attn_impl
        )
        
        # 5. Load fine-tuned weights
        # Note: If quantized, we load weights differently
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Filter state dict for the classifier head if needed
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[LocalGrader] Weights loaded from {model_path}")
        except Exception as e:
            print(f"[LocalGrader] Warning: Weight load failed or partial load: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[LocalGrader] Ready!")
    
    def grade_sync(self, request: GradeRequest) -> GradeResponse:
        """
        Grade an answer for hallucination (SYNCHRONOUS version).
        
        Target Latency: <15ms on T4 GPU
        
        Args:
            request: GradeRequest with context and answer
            
        Returns:
            GradeResponse with is_faithful, confidence, latency_ms
        """
        import time
        start = time.perf_counter()
        
        try:
            # Format input (matches training data format)
            input_text = f"Context: {request.context} Answer: {request.answer}"
            
            # Tokenize with ModernBERT's 8k context window
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                logits = self.model(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, predicted_class].item()
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            return GradeResponse(
                is_faithful=bool(predicted_class == 1),
                confidence=confidence,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            raise RuntimeError(f"[LocalGrader] Inference failed: {e}")


# Global instance for FastAPI startup
_grader_instance: Optional[LocalHallucinationGrader] = None


def get_grader() -> LocalHallucinationGrader:
    """Get or create the global grader instance."""
    global _grader_instance
    if _grader_instance is None:
        _grader_instance = LocalHallucinationGrader()
    return _grader_instance


def init_grader(model_path: str = "./models/guardrail_v1.pt"):
    """Initialize grader at FastAPI startup."""
    global _grader_instance
    _grader_instance = LocalHallucinationGrader(model_path=model_path)
    return _grader_instance
