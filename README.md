# ORBIT — Minecraft AI Agent System

ORBIT is a Python system that runs AI agents inside Minecraft: exploration, farming, building, and combat — coordinated by an event-driven orchestrator with a skill library, memory, vision, and local LLM support (Ollama).

It combines ideas from open-source projects (Agent Swarm, Voyager, Mindcraft / Mindcraft CE) into one setup. See commit history for details.

## Agents

| Agent | Role |
|-------|------|
| Explorer | Explores the world |
| Farmer | Farming tasks |
| Builder | Building tasks |
| Fighter | Combat tasks |

Base class: `agents/base.py`. Specialized agents: `agents/specialized.py`.

## How It Works

```
main.py
  -> core/config.py (OrbitConfig.from_env)
  -> core/orchestrator.py (initialize -> start -> stop)
  -> agents/ (Explorer, Farmer, Builder, Fighter)
  -> llm/ (Ollama client)
  -> memory/ | vision/ | skills/ | tasks/ | plugins/ | communication/
```

## Requirements

- Python 3.10+
- A running Minecraft server (Java) reachable at `MC_HOST:MC_PORT`
- Ollama running locally for LLM features (`OLLAMA_HOST`, default `http://localhost:11434`)

See `requirements.txt` for Python packages (`aiohttp`, `requests`, …).

## Setup

```bash
pip install -r requirements.txt
```

Configure via environment variables (see `core/config.py`):

```bash
# Linux / macOS
export OLLAMA_HOST="http://localhost:11434"
export MC_HOST="localhost"
export MC_PORT="25565"

# Windows (PowerShell)
$env:OLLAMA_HOST="http://localhost:11434"
$env:MC_HOST="localhost"
$env:MC_PORT="25565"
```

## Run

```bash
python main.py
# Press Ctrl+C to stop
```

## Test

`test.py` checks initialization, the Ollama connection, and a basic LLM chat:

```bash
python test.py
```

## Status

In development. Expect breaking changes. Contributions and issue reports are welcome.

## Credits

Built on ideas from the open-source Minecraft-agent community (Agent Swarm, Voyager, Mindcraft, Mindcraft CE). This repo is an independent integration — see those projects for the originals.
