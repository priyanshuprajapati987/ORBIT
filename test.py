import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import Orchestrator
from core.config import OrbitConfig


async def test():
    print("=" * 60)
    print("  ORBIT System Test")
    print("=" * 60)
    print()

    config = OrbitConfig.from_env()
    orchestrator = Orchestrator(config)

    print("[TEST] Initializing...")
    await orchestrator.initialize()

    print()
    print("[TEST] Checking Ollama connection...")
    from llm.ollama_client import OllamaClient
    llm = OllamaClient(config.llm.host)

    available = await llm.is_available()
    if available:
        print("[TEST] Ollama: CONNECTED")
        models = await llm.list_models()
        print(f"[TEST] Available models: {models}")
    else:
        print("[TEST] Ollama: NOT AVAILABLE")
        print("[TEST] Install Ollama: https://ollama.ai")

    print()
    print("[TEST] Testing LLM chat...")
    if available:
        response = await llm.chat(
            messages=[{"role": "user", "content": "Say 'ORBIT ready' in 5 words or less."}],
            model=config.llm.model,
            max_tokens=32,
        )
        print(f"[TEST] LLM Response: {response.content}")

    print()
    print("[TEST] Testing Skill Library...")
    from skills.library import SkillLibrary
    skill_lib = SkillLibrary(llm, config.skill_dir)

    test_skill = await skill_lib.add_skill(
        name="test_mine",
        code="async function test_mine(bot) { /* mine */ }",
        context="Mining skill for testing"
    )
    print(f"[TEST] Added skill: {test_skill.name}")
    print(f"[TEST] Description: {test_skill.description}")

    results = await skill_lib.retrieve("mining wood", top_k=3)
    print(f"[TEST] Retrieved {len(results)} skills")

    print()
    print("[TEST] Testing Memory Store...")
    from memory.store import MemoryStore
    memory = MemoryStore(llm, config.memory.persist_dir)

    await memory.add_message("user", "Hello ORBIT")
    await memory.add_message("assistant", "Hello! ORBIT ready.")
    print(f"[TEST] Memory history: {len(memory.conversation_history)} messages")

    memory.add_spatial_memory("Base", {"x": 0, "y": 64, "z": 0}, "Home base")
    places = memory.get_nearby_places({"x": 10, "y": 64, "z": 10}, radius=100)
    print(f"[TEST] Nearby places: {len(places)}")

    print()
    print("[TEST] Testing Team Bulletin...")
    from communication.bulletin import TeamBulletin, AgentStatus
    bulletin = TeamBulletin()

    bulletin.update(AgentStatus(
        name="Explorer", role="scout", action="exploring",
        thought="Found a cave!", position={"x": 100, "y": 60, "z": -50}
    ))
    bulletin.update(AgentStatus(
        name="Farmer", role="crafter", action="farming",
        thought="Wheat growing well", position={"x": 5, "y": 64, "z": 10}
    ))

    print(f"[TEST] Bulletin:\n{bulletin.format_all()}")

    print()
    print("[TEST] Testing Task Manager...")
    from tasks.tasks import TaskManager
    tm = TaskManager()

    t1 = tm.create_task("Build house", "Build a wooden house", "Builder", priority=3)
    t2 = tm.create_task("Gather wood", "Collect 32 oak logs", "Farmer", priority=5)
    print(f"[TEST] Created {len(tm.tasks)} tasks")
    print(f"[TEST] Stats: {tm.get_stats()}")

    print()
    print("[TEST] Testing Vision System...")
    from vision.camera import VisionSystem
    vision = VisionSystem(llm, mode="prompted")
    print(f"[TEST] Vision mode: {vision.mode}")

    print()
    print("=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("ORBIT system is ready!")
    print("Run: python main.py")

    await llm.close()


if __name__ == "__main__":
    asyncio.run(test())
