"""MLM: Modular Language Model training and evaluation framework."""

__version__ = "0.1.0"

from mlm.core.models import ModelRegistry
from mlm.core.trainer import Trainer
from mlm.core.evaluator import MTEBEvaluator

__all__ = ["ModelRegistry", "Trainer", "MTEBEvaluator"]