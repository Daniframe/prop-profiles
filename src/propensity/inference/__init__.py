"""Sector 3: instance-level model outcomes across incitement levels
(arXiv 2602.18182, stage 3).

- backends: InferenceBackend implementations (VLLMBackend, OpenAICompatibleBackend).
- models: vLLM model-shortcut registry.
- extraction: response post-processing (cleanup, <think> stripping, final-answer regex).
- runner: the shared prompt x question inference loop, backend-agnostic.
"""

from .backends import InferenceBackend, VLLMBackend, OpenAICompatibleBackend
from .models import MODEL_SHORTCUTS, THINKING_MODELS, resolve_model
from .extraction import clean_text, strip_thinking, extract_final_answer, FINAL_ANSWER_SUFFIX
from .runner import load_prompts, run_inference

__all__ = [
    "InferenceBackend",
    "VLLMBackend",
    "OpenAICompatibleBackend",
    "MODEL_SHORTCUTS",
    "THINKING_MODELS",
    "resolve_model",
    "clean_text",
    "strip_thinking",
    "extract_final_answer",
    "FINAL_ANSWER_SUFFIX",
    "load_prompts",
    "run_inference",
]
