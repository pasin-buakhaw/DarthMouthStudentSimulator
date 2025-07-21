from . import LLMClient
from anthropic import Anthropic



class AnthropicClient(LLMClient):
    """Anthropic Claude client"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    

