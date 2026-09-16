import json
import os
import hashlib
from typing import Optional
from dataclasses import dataclass, field
from llm.ollama_client import OllamaClient


@dataclass
class Skill:
    name: str
    code: str
    description: str
    metadata: dict = field(default_factory=dict)


class SkillLibrary:
    def __init__(self, llm: OllamaClient, persist_dir: str = "./orbit_skills"):
        self.llm = llm
        self.persist_dir = persist_dir
        self.skills: dict[str, Skill] = {}
        self.embeddings: dict[str, list[float]] = {}
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(os.path.join(self.persist_dir, "code"), exist_ok=True)
        os.makedirs(os.path.join(self.persist_dir, "descriptions"), exist_ok=True)

    def _skill_id(self, name: str) -> str:
        return hashlib.md5(name.encode()).hexdigest()[:12]

    async def add_skill(self, name: str, code: str, context: str = "") -> Skill:
        description = await self._generate_description(name, code, context)

        skill = Skill(name=name, code=code, description=description)
        self.skills[name] = skill

        embedding = await self.llm.embed(description)
        if embedding:
            self.embeddings[name] = embedding

        self._persist_skill(skill)
        return skill

    async def retrieve(self, query: str, top_k: int = 5) -> list[Skill]:
        if not self.embeddings:
            return []

        query_embedding = await self.llm.embed(query)
        if not query_embedding:
            return list(self.skills.values())[:top_k]

        scores = []
        for name, emb in self.embeddings.items():
            score = self._cosine_similarity(query_embedding, emb)
            scores.append((name, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for name, score in scores[:top_k]:
            if name in self.skills:
                results.append(self.skills[name])

        return results

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def list_skills(self) -> list[str]:
        return list(self.skills.keys())

    async def _generate_description(self, name: str, code: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "Generate a concise description of what this Minecraft skill does. 1-2 sentences max."},
            {"role": "user", "content": f"Function: {name}\nCode:\n{code}\n\n{context}"}
        ]

        response = await self.llm.chat(
            messages=messages,
            model="llama3.2",
            temperature=0.3,
            max_tokens=128,
        )

        return response.content.strip() if response.done else f"Skill: {name}"

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _persist_skill(self, skill: Skill):
        sid = self._skill_id(skill.name)

        code_path = os.path.join(self.persist_dir, "code", f"{sid}.js")
        with open(code_path, "w") as f:
            f.write(skill.code)

        desc_path = os.path.join(self.persist_dir, "descriptions", f"{sid}.txt")
        with open(desc_path, "w") as f:
            f.write(skill.description)

        index_path = os.path.join(self.persist_dir, "index.json")
        index = {}
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)

        index[skill.name] = {
            "id": sid,
            "description": skill.description,
            "metadata": skill.metadata,
        }

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def load_all(self):
        index_path = os.path.join(self.persist_dir, "index.json")
        if not os.path.exists(index_path):
            return

        with open(index_path, "r") as f:
            index = json.load(f)

        for name, info in index.items():
            sid = info["id"]
            code_path = os.path.join(self.persist_dir, "code", f"{sid}.js")
            if os.path.exists(code_path):
                with open(code_path, "r") as f:
                    code = f.read()
                self.skills[name] = Skill(
                    name=name,
                    code=code,
                    description=info.get("description", ""),
                    metadata=info.get("metadata", {}),
                )
