import asyncio
import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import deque


class EventType(IntEnum):
    REACTIVE = 0
    CRITIC = 1
    CHAT = 2
    STRATEGIC = 3
    SYSTEM = 4


@dataclass
class Event:
    type: EventType
    priority: int
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Callable] = None

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class EventQueue:
    def __init__(self, max_size: int = 100):
        self.queue: deque = deque()
        self.max_size = max_size
        self.processing = False
        self.cooldowns: dict[str, float] = {}
        self.dedup_window: dict[str, float] = {}

    def push(self, event: Event) -> bool:
        if self._is_duplicate(event):
            return False

        if len(self.queue) >= self.max_size:
            self.queue.pop()

        self.queue.append(event)
        self._sort_queue()
        return True

    def pop(self) -> Optional[Event]:
        if self.queue:
            return self.queue.popleft()
        return None

    def peek(self) -> Optional[Event]:
        if self.queue:
            return self.queue[0]
        return None

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def size(self) -> int:
        return len(self.queue)

    def clear(self):
        self.queue.clear()

    def _is_duplicate(self, event: Event) -> bool:
        key = f"{event.type}:{event.data.get('source', '')}"
        now = time.time()

        if key in self.dedup_window:
            if now - self.dedup_window[key] < 2.0:
                existing = self._find_by_key(key)
                if existing and event.priority >= existing.priority:
                    return True
                elif existing:
                    self.queue.remove(existing)

        self.dedup_window[key] = now
        self._cleanup_dedup()
        return False

    def _find_by_key(self, key: str) -> Optional[Event]:
        for event in self.queue:
            event_key = f"{event.type}:{event.data.get('source', '')}"
            if event_key == key:
                return event
        return None

    def _cleanup_dedup(self):
        now = time.time()
        expired = [k for k, v in self.dedup_window.items() if now - v > 5.0]
        for k in expired:
            del self.dedup_window[k]

    def _sort_queue(self):
        sorted_queue = sorted(self.queue)
        self.queue = deque(sorted_queue)


class Brain:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.event_queue = EventQueue()
        self.running = False
        self.paused = False
        self.handlers: dict[EventType, Callable] = {}
        self.strategic_cooldown = 8.0
        self.reactive_cooldown = 3.0
        self.last_strategic_time = 0
        self.last_reactive_time = 0

    def register_handler(self, event_type: EventType, handler: Callable):
        self.handlers[event_type] = handler

    async def push_event(self, event: Event) -> bool:
        if self.paused and event.type != EventType.REACTIVE:
            return False

        if not self.event_queue.push(event):
            return False

        if not self.processing:
            await self._process_next()
        return True

    async def _process_next(self):
        if self.processing or self.event_queue.is_empty():
            return

        self.processing = True
        event = self.event_queue.pop()

        if event.type == EventType.STRATEGIC:
            now = time.time()
            if now - self.last_strategic_time < self.strategic_cooldown:
                self.event_queue.push(event)
                self.processing = False
                return
            self.last_strategic_time = now

        elif event.type == EventType.REACTIVE:
            now = time.time()
            if now - self.last_reactive_time < self.reactive_cooldown:
                self.event_queue.push(event)
                self.processing = False
                return
            self.last_reactive_time = now

        handler = self.handlers.get(event.type)
        if handler:
            try:
                await handler(event)
            except Exception as e:
                print(f"[{self.agent_name}] Event handler error: {e}")

        self.processing = False

        if not self.event_queue.is_empty():
            await self._process_next()

    async def start(self):
        self.running = True
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        self.event_queue.clear()
