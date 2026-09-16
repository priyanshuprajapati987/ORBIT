import json
import time
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    name: str
    description: str
    assigned_to: str
    priority: int = 5
    status: str = "pending"
    goal: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[dict] = None


class TaskManager:
    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.validators: dict[str, Callable] = {}
        self.task_counter = 0

    def create_task(
        self,
        name: str,
        description: str,
        assigned_to: str,
        priority: int = 5,
        goal: dict = None,
    ) -> Task:
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        task = Task(
            id=task_id,
            name=name,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            goal=goal or {},
        )

        self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_agent_tasks(self, agent_name: str) -> list[Task]:
        return [
            t for t in self.tasks.values()
            if t.assigned_to == agent_name and t.status != "completed"
        ]

    def get_pending_tasks(self) -> list[Task]:
        return [
            t for t in self.tasks.values()
            if t.status == "pending"
        ]

    def complete_task(self, task_id: str, result: dict = None) -> bool:
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = "completed"
        task.completed_at = time.time()
        task.result = result
        return True

    def fail_task(self, task_id: str, reason: str = "") -> bool:
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = "failed"
        task.completed_at = time.time()
        task.result = {"error": reason}
        return True

    def register_validator(self, task_name: str, validator: Callable):
        self.validators[task_name] = validator

    def validate_task(self, task_id: str) -> dict:
        task = self.tasks.get(task_id)
        if not task:
            return {"valid": False, "error": "Task not found"}

        validator = self.validators.get(task.name)
        if validator:
            return validator(task)

        return {"valid": True, "score": 1.0}

    def reassign_task(self, task_id: str, new_agent: str) -> bool:
        task = self.tasks.get(task_id)
        if not task or task.status == "completed":
            return False

        task.assigned_to = new_agent
        return True

    def get_stats(self) -> dict:
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")
        pending = sum(1 for t in self.tasks.values() if t.status == "pending")
        in_progress = sum(1 for t in self.tasks.values() if t.status == "in_progress")

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "in_progress": in_progress,
            "completion_rate": completed / total if total > 0 else 0,
        }

    def auto_assign(self, agents: list[str]) -> list[Task]:
        pending = self.get_pending_tasks()
        assigned = []

        for i, task in enumerate(pending):
            agent = agents[i % len(agents)]
            task.assigned_to = agent
            task.status = "in_progress"
            assigned.append(task)

        return assigned

    def cleanup_old(self, max_age_hours: int = 24):
        now = time.time()
        cutoff = now - (max_age_hours * 3600)

        old_ids = [
            tid for tid, task in self.tasks.items()
            if task.status in ("completed", "failed") and task.created_at < cutoff
        ]

        for tid in old_ids:
            del self.tasks[tid]
