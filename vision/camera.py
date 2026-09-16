import os
import base64
import time
from typing import Optional
from llm.ollama_client import OllamaClient


class VisionSystem:
    def __init__(self, llm: OllamaClient, mode: str = "prompted", screenshot_dir: str = "./screenshots"):
        self.llm = llm
        self.mode = mode
        self.screenshot_dir = screenshot_dir
        self.last_screenshot: Optional[str] = None
        self.last_analysis: Optional[str] = None
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(self.screenshot_dir, exist_ok=True)

    async def capture(self, bot_state: dict = None) -> Optional[str]:
        if self.mode == "off":
            return None

        screenshot_path = os.path.join(
            self.screenshot_dir,
            f"screenshot_{int(time.time() * 1000)}.jpg"
        )

        self.last_screenshot = screenshot_path
        return screenshot_path

    async def analyze(self, image_path: str = None, context: str = "") -> str:
        path = image_path or self.last_screenshot
        if not path:
            return "No screenshot available."

        prompt = f"""Analyze this Minecraft screenshot.
What do you see? Describe:
- Blocks and terrain
- Mobs (friendly/hostile)
- Structures
- Items on ground
- Potential threats or resources

{context}

Be specific about block types and positions."""

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = await self.llm.chat(
            messages=messages,
            model="llava:7b",
            temperature=0.5,
            max_tokens=512,
            images=[base64.b64encode(open(path, "rb").read()).decode()] if os.path.exists(path) else None,
        )

        self.last_analysis = response.content
        return response.content

    async def detect_threats(self, context: str = "") -> list[dict]:
        if not self.last_analysis:
            return []

        prompt = f"""Based on this analysis: {self.last_analysis}
{context}

List any hostile mobs or threats. Return JSON array:
[{{"type": "mob_name", "danger_level": "low/medium/high", "direction": "..."}}]"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = await self.llm.chat(
            messages=messages,
            model="llama3.2",
            temperature=0.3,
            max_tokens=256,
            format_json=True,
        )

        try:
            import json
            threats = json.loads(response.content)
            return threats if isinstance(threats, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    async def find_resources(self, resource_type: str = "any", context: str = "") -> list[dict]:
        if not self.last_analysis:
            return []

        prompt = f"""Based on this analysis: {self.last_analysis}
{context}

Find {resource_type} resources. Return JSON array:
[{{"type": "block_name", "count": number, "distance": "near/mid/far"}}]"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = await self.llm.chat(
            messages=messages,
            model="llama3.2",
            temperature=0.3,
            max_tokens=256,
            format_json=True,
        )

        try:
            import json
            resources = json.loads(response.content)
            return resources if isinstance(resources, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def set_mode(self, mode: str):
        if mode in ("off", "prompted", "always"):
            self.mode = mode
