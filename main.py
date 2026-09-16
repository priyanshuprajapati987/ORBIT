import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import OrbitConfig
from core.orchestrator import Orchestrator


async def main():
    print("=" * 60)
    print("  ORBIT v1.0 - Unified AI Agent System for Minecraft")
    print("  Combining: Agent Swarm + Voyager + Mindcraft + Mindcraft CE")
    print("=" * 60)
    print()

    config = OrbitConfig.from_env()

    print(f"LLM Host: {config.llm.host}")
    print(f"Model: {config.llm.model}")
    print(f"Vision Model: {config.llm.vision_model}")
    print(f"MC Server: {config.mc_host}:{config.mc_port}")
    print()

    orchestrator = Orchestrator(config)

    try:
        await orchestrator.initialize()
        print()
        print("System ready! Starting agents...")
        print("(Press Ctrl+C to stop)")
        print()

        await orchestrator.start()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
