import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    model: str = "llama3.2"
    fast_model: str = "llama3.2"
    vision_model: str = "llava:7b"
    embedding_model: str = "nomic-embed-text"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.7
    max_tokens: int = 1024
    think: bool = False


@dataclass
class MemoryConfig:
    max_history: int = 50
    summary_chunk_size: int = 10
    memory_char_limit: int = 2000
    persist_dir: str = "./orbit_memory"


@dataclass
class VisionConfig:
    mode: str = "prompted"  # off, prompted, always
    screenshot_dir: str = "./screenshots"
    capture_interval: float = 1.0


@dataclass
class AgentConfig:
    name: str
    role: str
    personality: str
    leash_radius: int = 200
    allowed_actions: list = field(default_factory=list)
    allowed_skills: list = field(default_factory=list)
    keep_items: list = field(default_factory=list)
    priorities: str = ""
    season_goal: str = ""


@dataclass
class OrbitConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    agents: list = field(default_factory=list)
    mc_host: str = "localhost"
    mc_port: int = 25565
    mc_version: str = "1.20.4"
    skill_dir: str = "./skills"
    plugin_dir: str = "./plugins"

    @classmethod
    def from_env(cls) -> "OrbitConfig":
        config = cls()
        config.llm.host = os.getenv("OLLAMA_HOST", config.llm.host)
        config.mc_host = os.getenv("MC_HOST", config.mc_host)
        config.mc_port = int(os.getenv("MC_PORT", str(config.mc_port)))
        return config

    def add_default_agents(self):
        self.agents = [
            AgentConfig(
                name="Explorer",
                role="scout",
                personality="Fearless explorer. Names caves. Narrates discoveries.",
                leash_radius=500,
                allowed_actions=["explore", "find_blocks", "find_mobs", "navigate"],
                allowed_skills=["exploreUntil", "mineBlock", "craftItem"],
                priorities="Find new areas, name locations, report discoveries."
            ),
            AgentConfig(
                name="Farmer",
                role="crafter",
                personality="Nurturing farmer. Obsesses over farm layouts.",
                leash_radius=150,
                allowed_actions=["farm", "craft", "smelt", "breed", "cook"],
                allowed_skills=["build_farm", "craftItem", "smeltItem", "breed_animals"],
                priorities="Maintain farms, craft tools, manage resources."
            ),
            AgentConfig(
                name="Builder",
                role="builder",
                personality="Meticulous architect. Measures twice. Upset by asymmetry.",
                leash_radius=150,
                allowed_actions=["build", "place_blocks", "design"],
                allowed_skills=["build_house", "build_bridge", "placeBlock", "light_area"],
                priorities="Build structures, improve base, place decorations."
            ),
            AgentConfig(
                name="Fighter",
                role="guard",
                personality="Stoic warrior. Short sentences. Scans for threats.",
                leash_radius=300,
                allowed_actions=["combat", "defend", "patrol", "hunt"],
                allowed_skills=["neural_combat", "defendSelf", "attackEntity"],
                priorities="Defend base, hunt hostile mobs, protect teammates."
            ),
        ]
