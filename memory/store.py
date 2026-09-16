import json
import os
import time
from typing import Optional
from dataclasses import dataclass, field
from llm.ollama_client import OllamaClient


@dataclass
class MemoryEntry:
    content: str
    timestamp: float
    memory_type: str = "conversation"
    importance: float = 0.5
    metadata: dict = field(default_factory=dict)


class MemoryStore:
    def __init__(self, llm: OllamaClient, persist_dir: str = "./orbit_memory"):
        self.llm = llm
        self.persist_dir = persist_dir
        self.conversation_history: list[dict] = []
        self.memories: list[MemoryEntry] = []
        self.spatial_memory: dict[str, dict] = {}
        self.max_history = 50
        self.memory_char_limit = 2000
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(self.persist_dir, exist_ok=True)

    async def add_message(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

        if len(self.conversation_history) > self.max_history:
            chunk = self.conversation_history[:10]
            self.conversation_history = self.conversation_history[10:]
            await self._summarize_chunk(chunk)

    async def _summarize_chunk(self, chunk: list[dict]):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in chunk)

        messages = [
            {"role": "system", "content": "Summarize this conversation in 1-2 sentences. Be concise."},
            {"role": "user", "content": text}
        ]

        response = await self.llm.chat(
            messages=messages,
            model="llama3.2",
            temperature=0.3,
            max_tokens=128,
        )

        if response.done:
            summary = response.content.strip()
            self.memories.append(MemoryEntry(
                content=summary,
                timestamp=time.time(),
                memory_type="summary",
                importance=0.7,
            ))

    def get_memory_context(self) -> str:
        if not self.memories:
            return ""

        recent = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:5]
        return "\n".join(f"- {m.content}" for m in recent)

    def get_conversation_history(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.conversation_history]

    def add_spatial_memory(self, name: str, position: dict, description: str = ""):
        self.spatial_memory[name] = {
            "position": position,
            "description": description,
            "timestamp": time.time(),
        }

    def get_nearby_places(self, position: dict, radius: int = 100) -> list[dict]:
        results = []
        for name, data in self.spatial_memory.items():
            dx = data["position"]["x"] - position["x"]
            dz = data["position"]["z"] - position["z"]
            dist = (dx * dx + dz * dz) ** 0.5
            if dist <= radius:
                results.append({"name": name, "distance": dist, **data})
        return sorted(results, key=lambda x: x["distance"])

    def save(self):
        data = {
            "memories": [
                {
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "memory_type": m.memory_type,
                    "importance": m.importance,
                    "metadata": m.metadata,
                }
                for m in self.memories
            ],
            "spatial_memory": self.spatial_memory,
        }

        path = os.path.join(self.persist_dir, "memory.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        path = os.path.join(self.persist_dir, "memory.json")
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            data = json.load(f)

        self.memories = [
            MemoryEntry(
                content=m["content"],
                timestamp=m["timestamp"],
                memory_type=m.get("memory_type", "conversation"),
                importance=m.get("importance", 0.5),
                metadata=m.get("metadata", {}),
            )
            for m in data.get("memories", [])
        ]
        self.spatial_memory = data.get("spatial_memory", {})
