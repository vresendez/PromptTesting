import os
import openai
import asyncio
import fastapi_poe as fp
from dotenv import load_dotenv
    
load_dotenv()
class LLMClient:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.poe_api_key = os.getenv("POE_API_KEY")

    def gpt_call(self, prompt, model="gpt-4-32k-0613", temperature=0):
        client = openai.OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def llama_call(self, prompt):
        return f"LLaMA response to: {prompt}"

    def gemini_call(self, prompt):
        return f"Gemini response to: {prompt}"

    def claude_call(self, prompt):
        return f"Claude response to: {prompt}"

    def deepseek_call(self, prompt):
        return f"DeepSeek response to: {prompt}"

    async def _poe_call_async(self, prompt, bot_name="GPT-4o"):
        messages = [fp.ProtocolMessage(role="user", content=prompt)]
        response_text = ""
        async for part in fp.get_bot_response(messages=messages, bot_name=bot_name, api_key=self.poe_api_key):
            response_text += part.text
        return response_text

    def poe_call(self, prompt, bot_name="GPT-4o"):
        return asyncio.run(self._poe_call_async(prompt, bot_name))
