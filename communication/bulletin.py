import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentStatus:
    name: str
    role: str
    action: str
    thought: str
    position: dict = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    health: int = 20
    hunger: int = 20
    timestamp: float = field(default_factory=time.time)


class TeamBulletin:
    def __init__(self):
        self.statuses: dict[str, AgentStatus] = {}
        self.stale_threshold = 30.0

    def update(self, status: AgentStatus):
        self.statuses[status.name] = status

    def get(self, name: str) -> Optional[AgentStatus]:
        return self.statuses.get(name)

    def get_all(self) -> list[AgentStatus]:
        return list(self.statuses.values())

    def format_for(self, exclude_name: str) -> str:
        teammates = [
            s for s in self.statuses.values()
            if s.name != exclude_name
        ]

        if not teammates:
            return ""

        lines = []
        for t in teammates:
            pos = f"({t.position['x']}, {t.position['y']}, {t.position['z']})"
            age = round(time.time() - t.timestamp)
            stale = " [stale]" if age > self.stale_threshold else ""
            lines.append(f"- {t.name}: {t.action} at {pos} - \"{t.thought}\"{stale}")

        return "\n".join(lines)

    def format_all(self) -> str:
        lines = []
        for t in self.statuses.values():
            pos = f"({t.position['x']}, {t.position['y']}, {t.position['z']})"
            lines.append(f"{t.name} [{t.role}]: {t.action} at {pos}")
        return "\n".join(lines)

    def get_status_summary(self) -> dict:
        summary = {}
        for name, status in self.statuses.items():
            age = time.time() - status.timestamp
            summary[name] = {
                "role": status.role,
                "action": status.action,
                "position": status.position,
                "health": status.health,
                "age_seconds": round(age),
                "stale": age > self.stale_threshold,
            }
        return summary

    def clear_stale(self):
        now = time.time()
        stale_names = [
            name for name, status in self.statuses.items()
            if now - status.timestamp > self.stale_threshold * 2
        ]
        for name in stale_names:
            del self.statuses[name]
