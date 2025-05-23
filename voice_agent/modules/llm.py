from openai import OpenAI
from livekit.agents.llm import LLM

class OpenRouterLLM(LLM):
    def __init__(self,api_key,base_url):
        self.client = OpenAI(
          base_url=base_url,
          api_key=api_key,
        )
        self.history = [
            {"role": "system", "content": "You are a helpful voice assistant."}
        ]

    def chat(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(
          model="deepseek/deepseek-chat-v3-0324:free",
          messages=self.history
        )
        return completion.choices[0].message.content