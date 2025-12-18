from abc import ABC, abstractmethod
import torch

class BaseNeutralizer(ABC):
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Initializing on {self.device}...")

    @abstractmethod
    def debias(self, text: str) -> str:
        """Takes a single biased string, returns the neutral version."""
        pass

    def batch_debias(self, texts: list) -> list:
        """Optional: Optimized batch processing."""
        return [self.debias(t) for t in texts]