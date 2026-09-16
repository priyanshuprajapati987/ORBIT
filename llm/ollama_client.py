import json
import asyncio
import aiohttp
from typing import Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    thinking: Optional[str] = None
    done: bool = True


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434"):
        self.host = host.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def chat(
        self,
        messages: list[dict],
        model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        think: bool = False,
        format_json: bool = False,
        images: Optional[list[str]] = None,
    ) -> LLMResponse:
        session = await self._get_session()

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if think:
            payload["think"] = True

        if format_json:
            payload["format"] = "json"

        if images:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    msg["images"] = images
                    break

        try:
            async with session.post(
                f"{self.host}/api/chat", json=payload
            ) as resp:
                data = await resp.json()

                content = data.get("message", {}).get("content", "")
                thinking = None

                if think and "<thinking>" in content and "</thinking>" in content:
                    parts = content.split("</thinking>")
                    thinking = parts[0].replace("<thinking>", "").strip()
                    content = parts[1].strip() if len(parts) > 1 else ""

                return LLMResponse(
                    content=content,
                    model=model,
                    thinking=thinking,
                    done=data.get("done", True),
                )
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=model,
                done=False,
            )

    async def embed(
        self, text: str, model: str = "nomic-embed-text"
    ) -> Optional[list[float]]:
        session = await self._get_session()

        try:
            async with session.post(
                f"{self.host}/api/embeddings",
                json={"model": model, "input": text},
            ) as resp:
                data = await resp.json()
                return data.get("embedding")
        except Exception:
            return None

    async def list_models(self) -> list[str]:
        session = await self._get_session()
        try:
            async with session.get(f"{self.host}/api/tags") as resp:
                data = await resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def is_available(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self.host}/api/tags") as resp:
                return resp.status == 200
        except Exception:
            return False
