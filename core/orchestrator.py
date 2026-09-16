import asyncio
import time
from typing import Optional
from core.config import OrbitConfig
from core.events import Event, EventType
from llm.ollama_client import OllamaClient
from agents.base import BaseAgent
from agents.specialized import ExplorerAgent, FarmerAgent, BuilderAgent, FighterAgent
from skills.library import SkillLibrary
from memory.store import MemoryStore
from communication.bulletin import TeamBulletin
from vision.camera import VisionSystem
from tasks.tasks import TaskManager
from plugins.base import PluginManager


class CurriculumAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm
        self.progress = 0
        self.completed_tasks = []
        self.failed_tasks = []
        self.knowledge_cache: dict[str, str] = {}

    async def propose_next_task(self, context: str, inventory: dict) -> tuple[str, str]:
        if self.progress == 0:
            return "Mine 1 wood log", "Start by gathering basic resources."

        messages = [
            {"role": "system", "content": """You are a curriculum agent for Minecraft.
Propose the next task based on current state.
Consider: inventory, biome, progress, team needs.
Return JSON: {"task": "...", "context": "...", "reasoning": "..."}"""},
            {"role": "user", "content": f"Context:\n{context}\n\nInventory: {inventory}\n\nCompleted: {self.completed_tasks[-5:]}\n\nWhat should be next?"}
        ]

        response = await self.llm.chat(
            messages=messages,
            model="llama3.2",
            temperature=0.7,
            max_tokens=256,
            format_json=True,
        )

        try:
            import json
            result = json.loads(response.content)
            return result.get("task", "Explore"), result.get("context", "")
        except (json.JSONDecodeError, TypeError):
            return "Explore surroundings", "Look for resources and dangers."

    def update_progress(self, success: bool, task: str):
        self.progress += 1
        if success:
            self.completed_tasks.append(task)
        else:
            self.failed_tasks.append(task)


class CriticAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    async def evaluate(self, action: str, result: str, context: str) -> dict:
        messages = [
            {"role": "system", "content": """Evaluate if the task was successful.
Return JSON: {"success": bool, "score": 0.0-1.0, "critique": "..."}"""},
            {"role": "user", "content": f"Action: {action}\nResult: {result}\nContext:\n{context}"}
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
            return json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            return {"success": False, "score": 0, "critique": "Could not evaluate"}


class Orchestrator:
    def __init__(self, config: Optional[OrbitConfig] = None):
        self.config = config or OrbitConfig()
        if not self.config.agents:
            self.config.add_default_agents()

        self.llm = OllamaClient(host=self.config.llm.host)
        self.bulletin = TeamBulletin()
        self.memory = MemoryStore(self.llm, self.config.memory.persist_dir)
        self.skill_library = SkillLibrary(self.llm, self.config.skill_dir)
        self.vision = VisionSystem(self.llm, self.config.vision.mode)
        self.task_manager = TaskManager()
        self.plugin_manager = PluginManager(self.config.plugin_dir)

        self.agents: dict[str, BaseAgent] = {}
        self.curriculum = CurriculumAgent(self.llm)
        self.critic = CriticAgent(self.llm)

        self.running = False
        self.start_time = 0

    async def initialize(self):
        print("[ORBIT] Initializing system...")

        available = await self.llm.is_available()
        if not available:
            print("[ORBIT] WARNING: Ollama not available. LLM features will be limited.")
        else:
            models = await self.llm.list_models()
            print(f"[ORBIT] Ollama models: {models}")

        self.memory.load()
        self.skill_library.load_all()

        agent_classes = {
            "scout": ExplorerAgent,
            "crafter": FarmerAgent,
            "builder": BuilderAgent,
            "guard": FighterAgent,
        }

        for agent_config in self.config.agents:
            agent_class = agent_classes.get(agent_config.role, BaseAgent)
            agent = agent_class(self.llm, self.bulletin)
            self.agents[agent_config.name] = agent
            print(f"[ORBIT] Created agent: {agent_config.name} ({agent_config.role})")

        await self.plugin_manager.load_all()
        print(f"[ORBIT] Loaded {len(self.plugin_manager.plugins)} plugins")

        print("[ORBIT] Initialization complete!")

    async def start(self):
        if self.running:
            return

        self.running = True
        self.start_time = time.time()
        print("[ORBIT] Starting agent swarm...")

        for agent in self.agents.values():
            await agent.start()

        await self._main_loop()

    async def stop(self):
        self.running = False
        print("[ORBIT] Stopping...")

        for agent in self.agents.values():
            await agent.stop()

        self.memory.save()
        await self.llm.close()
        print("[ORBIT] Stopped.")

    async def _main_loop(self):
        cycle = 0
        while self.running:
            cycle += 1
            elapsed = time.time() - self.start_time

            print(f"\n[ORBIT] === Cycle {cycle} ({elapsed:.0f}s) ===")
            print(self.bulletin.format_all())

            context = self._build_global_context()

            task, task_context = await self.curriculum.propose_next_task(
                context, self._get_total_inventory()
            )

            print(f"[ORBIT] Curriculum proposes: {task}")

            for agent_name, agent in self.agents.items():
                await agent.push_event(Event(
                    type=EventType.STRATEGIC,
                    priority=5,
                    data={"task": task, "context": task_context},
                ))

            await asyncio.sleep(0.1)

            for agent_name, agent in self.agents.items():
                await agent.push_event(Event(
                    type=EventType.CRITIC,
                    priority=3,
                    data={
                        "action": agent.current_action,
                        "result": "cycle completed",
                    },
                ))

            if cycle % 10 == 0:
                self.bulletin.clear_stale()
                self.memory.save()

            await asyncio.sleep(5)

    def _build_global_context(self) -> str:
        ctx = "GLOBAL STATE:\n"
        ctx += f"Runtime: {time.time() - self.start_time:.0f}s\n"
        ctx += f"Agents: {', '.join(self.agents.keys())}\n"
        ctx += f"Skills learned: {len(self.skill_library.skills)}\n"
        ctx += f"Tasks completed: {self.curriculum.progress}\n"

        team = self.bulletin.format_all()
        if team:
            ctx += f"\nTEAM:\n{team}\n"

        memory_ctx = self.memory.get_memory_context()
        if memory_ctx:
            ctx += f"\nRECENT MEMORIES:\n{memory_ctx}\n"

        return ctx

    def _get_total_inventory(self) -> dict:
        total = {}
        for agent in self.agents.values():
            for item, count in agent.inventory.items():
                total[item] = total.get(item, 0) + count
        return total

    def update_agent_state(
        self,
        agent_name: str,
        position: dict = None,
        health: int = None,
        hunger: int = None,
        inventory: dict = None,
    ):
        agent = self.agents.get(agent_name)
        if agent:
            agent.update_state(position, health, hunger, inventory)

    def get_stats(self) -> dict:
        uptime = time.time() - self.start_time if self.start_time else 0
        return {
            "uptime_seconds": uptime,
            "agents": list(self.agents.keys()),
            "skills": len(self.skill_library.skills),
            "tasks": self.task_manager.get_stats(),
            "curriculum_progress": self.curriculum.progress,
            "bulletin_status": self.bulletin.get_status_summary(),
        }
