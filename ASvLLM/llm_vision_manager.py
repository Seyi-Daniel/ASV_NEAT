import os
import re
import json
import pygame
import base64
from typing import Optional

# Placeholder for fetch, same as before
try:
    from js import fetch
except ImportError:
    print("Warning: 'js.fetch' not available. Using a placeholder. Install 'requests' for local use.")
    import requests
    from types import SimpleNamespace


    async def fetch(url, options):
        real_response = requests.post(url, headers=options.get('headers'), data=options.get('body'))
        mock_response = SimpleNamespace()
        mock_response.ok = real_response.status_code == 200
        mock_response.status = real_response.status_code
        _text = real_response.text

        def _json():
            try:
                return real_response.json()
            except json.JSONDecodeError:
                return {}

        async def text():
            return _text

        async def json():
            return _json()

        mock_response.text = text
        mock_response.json = json
        return mock_response

# --- API Key Environment Variables ---
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV = "CLAUDE_API_KEY"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"


class LLMVisionManager:
    """
    Handles sending a snapshot + prompt to a multimodal LLM.
    It can be initialized to use OpenAI, Claude, or DeepSeek.
    """

    def __init__(self, provider: str = "openai"):
        print(f"Initializing LLMVisionManager for provider: {provider}...")
        self.provider = provider
        self.api_key = self._get_api_key(provider)
        if not self.api_key:
            raise ValueError(f"FATAL: API key for {provider} is not set.")
        print(f"Vision Manager for {provider} is ready.")

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Fetches the correct API key from environment variables."""
        if provider == "openai":
            return os.getenv(OPENAI_API_KEY_ENV)
        elif provider == "claude":
            return os.getenv(ANTHROPIC_API_KEY_ENV)
        elif provider == "deepseek":
            return os.getenv(DEEPSEEK_API_KEY_ENV)
        return None

    def _clean_json_response(self, content: str) -> Optional[str]:
        """Finds and extracts a JSON array from the LLM's messy output."""
        if not isinstance(content, str): return None
        match = re.search(r'```(json)?\s*(\[.*\])\s*```', content, re.DOTALL)
        if match: return match.group(2)
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1 and end > start: return content[start:end + 1]
        return None

    async def get_llm_decision_from_image(self, screen_surface: pygame.Surface, prompt: str) -> Optional[str]:
        """
        Takes a pygame surface, converts it, and routes to the correct API.
        """
        print(f"\n--- Vision Manager ({self.provider}): Processing image for LLM... ---")
        try:
            # Convert surface to base64
            image_filename = "temp_for_api.jpg"
            pygame.image.save(screen_surface, image_filename)
            with open(image_filename, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Route to the correct provider's method
            if self.provider == "openai":
                return await self._call_openai_vision(prompt, base64_image)
            elif self.provider == "claude":
                return await self._call_claude_vision(prompt, base64_image)
            elif self.provider == "deepseek":
                return await self._call_deepseek_vision(prompt, base64_image)
            else:
                print(f"Error: Unknown vision provider '{self.provider}'")
                return "[]"

        except Exception as e:
            print(f"Error during Vision API call execution: {e}")
            return "[]"

    # --- Provider-Specific Call Methods ---

    async def _call_openai_vision(self, prompt: str, base64_image: str) -> Optional[str]:
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o", "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}]
        }

        print(f"--- DATA SENT TO OpenAI ---")
        print(f"Image File: temp_for_api.jpg")
        print("Text Prompt Sent:\n", prompt)

        response = await fetch(api_url, {"method": 'POST', "headers": headers, "body": json.dumps(payload)})
        if not response.ok:
            error_text = await response.text();
            print(f"--- OpenAI API Error: {response.status} {error_text} ---");
            return "[]"
        result = await response.json()
        raw_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._clean_json_response(raw_text)

    async def _call_claude_vision(self, prompt: str, base64_image: str) -> Optional[str]:
        api_url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "claude-3-opus-20240229", "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                {"type": "text", "text": prompt}
            ]}]
        }

        print(f"--- DATA SENT TO Claude ---")
        print(f"Image File: temp_for_api.jpg")
        print("Text Prompt Sent:\n", prompt)

        response = await fetch(api_url, {"method": 'POST', "headers": headers, "body": json.dumps(payload)})
        if not response.ok:
            error_text = await response.text();
            print(f"--- Claude API Error: {response.status} {error_text} ---");
            return "[]"
        result = await response.json()
        raw_text = result.get("content", [{}])[0].get("text", "")
        return self._clean_json_response(raw_text)

    async def _call_deepseek_vision(self, prompt: str, base64_image: str) -> Optional[str]:
        api_url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-vl-chat", "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}]
        }

        print(f"--- DATA SENT TO DeepSeek ---")
        print(f"Image File: temp_for_api.jpg")
        print("Text Prompt Sent:\n", prompt)

        response = await fetch(api_url, {"method": 'POST', "headers": headers, "body": json.dumps(payload)})
        if not response.ok:
            error_text = await response.text();
            print(f"--- DeepSeek API Error: {response.status} {error_text} ---");
            return "[]"
        result = await response.json()
        raw_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._clean_json_response(raw_text)