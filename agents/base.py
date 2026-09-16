import asyncio
import time
from typing import Optional
from core.events import Brain, Event, EventType
from core.config import AgentConfig
from llm.ollama_client import OllamaClient
from communication.bulletin import TeamBulletin, AgentStatus


class BaseAgent:
    def __init__(self, config: AgentConfig, llm: OllamaClient, bulletin: TeamBulletin):
        self.config = config
        self.llm = llm
        self.bulletin = bulletin
        self.brain = Brain(config.name)
        self.position = {"x": 0, "y": 0, "z": 0}
        self.health = 20
        self.hunger = 20
        self.inventory: dict[str, int] = {}
        self.current_action = "idle"
        self.last_action_time = 0
        self.running = False
        self._setup_handlers()

    def _setup_handlers(self):
        self.brain.register_handler(EventType.REACTIVE, self._handle_reactive)
        self.brain.register_handler(EventType.STRATEGIC, self._handle_strategic)
        self.brain.register_handler(EventType.CHAT, self._handle_chat)
        self.brain.register_handler(EventType.CRITIC, self._handle_critic)

    def _build_context(self) -> str:
        ctx = f"Name: {self.config.name}\n"
        ctx += f"Role: {self.config.role}\n"
        ctx += f"Position: ({self.position['x']}, {self.position['y']}, {self.position['z']})\n"
        ctx += f"Health: {self.health}/20\n"
        ctx += f"Hunger: {self.hunger}/20\n"
        ctx += f"Current Action: {self.current_action}\n"

        if self.inventory:
            items = ", ".join(f"{k}:{v}" for k, v in self.inventory.items())
            ctx += f"Inventory: {items}\n"

        team_status = self.bulletin.format_for(self.config.name)
        if team_status:
            ctx += f"\nTEAM STATUS:\n{team_status}\n"

        if self.config.priorities:
            ctx += f"\nPriorities: {self.config.priorities}\n"

        if self.config.season_goal:
            ctx += f"Current Goal: {self.config.season_goal}\n"

        return ctx

    async def _handle_reactive(self, event: Event):
        situation = event.data.get("situation", "unknown threat")
        context = self._build_context()

        messages = [
            {"role": "system", "content": f"""You are {self.config.name}, a {self.config.role} bot.
React immediately to threats. Keep response short (1-2 sentences).
Actions: {', '.join(self.config.allowed_actions)}"""},
            {"role": "user", "content": f"Situation: {situation}\n\nContext:\n{context}\n\nWhat do you do?"}
        ]

        response = await self.llm.chat(
            messages=messages,
            model="llama3.2",
            temperature=0.3,
            max_tokens=256,
        )

        await self._update_bulletin(f"Reacting to {situation}", response.content)

    async def _handle_strategic(self, event: Event):
        context = self._build_context()

        messages = [
            {"role": "system", "content": f"""You are {self.config.name}, a {self.config.role} bot.
Choose your next action based on your role and priorities.
Available actions: {', '.join(self.config.allowed_actions)}
Available skills: {', '.join(self.config.allowed_skills)}
Respond with JSON: {{"action": "...", "reasoning": "...", "target": "..."}}"""},
            {"role": "user", "content": f"Context:\n{context}\n\nWhat should you do next?"}
        ]

        response = await self.llm.chat(
            messages=messages,
            model="llama3.2",
            temperature=0.7,
            max_tokens=512,
            format_json=True,
        )

        try:
            import json
            decision = json.loads(response.content)
            await self._execute_action(decision)
        except json.JSONDecodeError:
            await self._update_bulletin("Planning", response.content)

    async def _handle_chat(self, event: Event):
        message = event.data.get("message", "")
        sender = event.data.get("sender", "unknown")

        messages = [
            {"role": "system", "content": f"""You are {self.config.name}, a {self.config.role}.
Personality: {self.config.personality}
Respond naturally to: {sender}"""},
            {"role": "user", "content": message}
        ]

        response = await self.llm.chat(
            messages=messages,
            model="llama3.2",
            temperature=0.8,
            max_tokens=256,
        )

        await self._update_bulletin(f"Chatting with {sender}", response.content)

    async def _handle_critic(self, event: Event):
        action = event.data.get("action", "unknown")
        result = event.data.get("result", "unknown")
        context = self._build_context()

        messages = [
            {"role": "system", "content": "Evaluate if the action was successful. Return JSON: {\"success\": bool, \"critique\": \"...\"}"},
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
            verdict = json.loads(response.content)
            if not verdict.get("success", False):
                critique = verdict.get("critique", "Try again")
                await self.push_event(Event(
                    type=EventType.STRATEGIC,
                    priority=2,
                    data={"situation": f"Previous action failed: {critique}"}
                ))
        except json.JSONDecodeError:
            pass

    async def _execute_action(self, decision: dict):
        action = decision.get("action", "idle")
        self.current_action = action
        self.last_action_time = time.time()

        await self._update_bulletin(action, decision.get("reasoning", ""))

    async def _update_bulletin(self, action: str, thought: str):
        status = AgentStatus(
            name=self.config.name,
            role=self.config.role,
            action=action,
            thought=thought,
            position=self.position.copy(),
            health=self.health,
            hunger=self.hunger,
            timestamp=time.time(),
        )
        self.bulletin.update(status)

    async def push_event(self, event: Event):
        await self.brain.push_event(event)

    async def start(self):
        self.running = True
        await self.brain.start()
        await self._update_bulletin("Starting up", "Agent initialized")

    async def stop(self):
        self.running = False
        self.brain.stop()

    def update_state(self, position=None, health=None, hunger=None, inventory=None):
        if position:
            self.position = position
        if health is not None:
            self.health = health
        if hunger is not None:
            self.hunger = hunger
        if inventory is not None:
            self.inventory = inventory
