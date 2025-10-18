from .hello import router as hello
from .inference import router as inference
from .predict import router as predict

__all__ = ["hello", "inference", "predict"]
