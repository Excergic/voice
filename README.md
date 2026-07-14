# Voice Call Agent

A local voice AI agent for HVAC dispatch, built with [Pipecat](https://github.com/pipecat-ai/pipecat). Speak into your mic, the agent listens, responds out loud, and holds a full spoken conversation.

**Pipeline:** mic → Deepgram STT → GPT-4o-mini → Cartesia TTS → speakers

## What it does

- Greets the caller and collects: name, phone, address, equipment type, and symptom
- Classifies urgency (emergency / urgent / routine) and offers an appointment window
- Speaks in short natural sentences — no lists, no markdown
- Waits for you to finish speaking before responding (no echo self-interruption)

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- API keys for Deepgram, OpenAI, and Cartesia

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/Excergic/voice.git
cd voice

# 2. Install dependencies
uv sync

# 3. Create your .env file and add your API keys
cp .env.example .env
```

Edit `.env`:
```
DEEPGRAM_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
CARTESIA_API_KEY=your_key_here
```

Get your keys:
- Deepgram: https://console.deepgram.com (API Keys)
- OpenAI: https://platform.openai.com/api-keys
- Cartesia: https://play.cartesia.ai (API Keys)

## Run

```bash
uv run python -m app.main
```

Press `Ctrl+C` to end the call.

## Project structure

```
app/
  main.py    # pipeline wiring — mic, STT, LLM, TTS, speakers
  agent.py   # system prompt and CallState dataclass
.env.example # API key template
pyproject.toml
```

## Key design decisions

**Echo prevention (`AlwaysUserMuteStrategy`):** Local audio (speakers + mic in the same room) causes the mic to pick up the speaker output. Without muting, the VAD detects the agent's own voice as "user speaking" and fires a barge-in interruption every ~600ms, cutting every sentence short. `AlwaysUserMuteStrategy` suppresses mic input while the bot is speaking, then reopens it when the bot finishes.

**VAD tuning:** `stop_secs=0.8` — the default 200ms silence threshold is too aggressive for natural speech. 800ms gives you room to pause mid-thought without the system cutting you off.

## Tech stack

| Component | Service | Why |
|-----------|---------|-----|
| STT | Deepgram Nova | Low-latency streaming transcription |
| LLM | GPT-4o-mini | Fast, cheap, good at tool calling |
| TTS | Cartesia Sonic | Streaming TTS with sub-200ms first chunk |
| VAD | Silero (ONNX) | Runs locally, no extra API |
| Framework | Pipecat 1.5 | Handles the audio pipeline and turn-taking |
