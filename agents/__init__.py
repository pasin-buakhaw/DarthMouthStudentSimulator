
from abc import ABC, abstractmethod#,staticmethod

class LLMClient(ABC):
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        pass
    
    
    