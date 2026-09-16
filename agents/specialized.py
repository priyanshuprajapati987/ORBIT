from .base import BaseAgent
from core.config import AgentConfig
from llm.ollama_client import OllamaClient
from communication.bulletin import TeamBulletin


class ExplorerAgent(BaseAgent):
    def __init__(self, llm: OllamaClient, bulletin: TeamBulletin):
        config = AgentConfig(
            name="Explorer",
            role="scout",
            personality="Fearless explorer. Names caves. Narrates discoveries like a nature documentary host.",
            leash_radius=500,
            allowed_actions=["explore", "find_blocks", "find_mobs", "navigate", "map_area"],
            allowed_skills=["exploreUntil", "mineBlock", "craftItem"],
            priorities="Find new areas, name locations, report resource deposits to team.",
        )
        super().__init__(config, llm, bulletin)


class FarmerAgent(BaseAgent):
    def __init__(self, llm: OllamaClient, bulletin: TeamBulletin):
        config = AgentConfig(
            name="Farmer",
            role="crafter",
            personality="Nurturing farmer. Names animals. Obsesses over farm symmetry and efficiency.",
            leash_radius=150,
            allowed_actions=["farm", "craft", "smelt", "breed", "cook", "manage_crops"],
            allowed_skills=["build_farm", "craftItem", "smeltItem", "breed_animals", "useChest"],
            priorities="Maintain farms, craft tools, manage resources, keep food supply high.",
        )
        super().__init__(config, llm, bulletin)


class BuilderAgent(BaseAgent):
    def __init__(self, llm: OllamaClient, bulletin: TeamBulletin):
        config = AgentConfig(
            name="Builder",
            role="builder",
            personality="Meticulous architect. Measures twice. Gets upset by asymmetry and bad design.",
            leash_radius=150,
            allowed_actions=["build", "place_blocks", "design", "demolish", "light_area"],
            allowed_skills=["build_house", "build_bridge", "placeBlock", "light_area", "setup_stash"],
            priorities="Build structures, improve base, ensure symmetry and good lighting.",
        )
        super().__init__(config, llm, bulletin)


class FighterAgent(BaseAgent):
    def __init__(self, llm: OllamaClient, bulletin: TeamBulletin):
        config = AgentConfig(
            name="Fighter",
            role="guard",
            personality="Stoic warrior. Short sentences. Constantly scans for threats.",
            leash_radius=300,
            allowed_actions=["combat", "defend", "patrol", "hunt", "scout_danger"],
            allowed_skills=["neural_combat", "defendSelf", "attackEntity", "craft_gear"],
            priorities="Defend base, hunt hostile mobs, protect teammates, patrol perimeter.",
        )
        super().__init__(config, llm, bulletin)
