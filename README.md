# Voice Call Agent — v1 (Twilio)

An inbound HVAC dispatch agent that answers real phone calls via Twilio, built with [Pipecat](https://github.com/pipecat-ai/pipecat).

A caller dials your Twilio number → Twilio streams audio to this server → the agent listens, speaks, and follows a structured five-stage conversation.

## Architecture

```
Caller (Phone)
      │
      ▼
Twilio (PSTN Gateway)   ←────────────────────────┐
      │                                           │
      ▼                                           │
Pipecat (Orchestrator)                            │
  ├── Deepgram (STT)   ← transcribes caller audio │
  ├── GPT-4o-mini (LLM) ← generates responses    │
  └── Cartesia (TTS)   ─────────────────────────►┘
```

**Transport:** Twilio Media Streams over WebSocket. Audio is μ-law 8kHz on the wire; Pipecat converts to PCM internally.

## Conversation stages

Every call moves through five stages in order:

| Stage | What happens |
|-------|-------------|
| **Greeting** | Agent introduces itself and asks how it can help |
| **Triage** | Determines urgency: emergency / urgent / routine |
| **Info Gathering** | Collects name, callback number, address, equipment type, symptom |
| **Confirmation** | Reads back details, offers appointment window, confirms booking |
| **Goodbye** | Thanks caller, summarizes next steps, closes the call |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A [Twilio account](https://console.twilio.com) with a phone number
- A public HTTPS URL (use [ngrok](https://ngrok.com) for local dev)
- API keys for Deepgram, OpenAI, and Cartesia

## Setup

```bash
# 1. Clone and switch to v1-branch
git clone https://github.com/Excergic/voice.git
cd voice
git checkout v1-branch

# 2. Install dependencies
uv sync

# 3. Create .env and add your keys
cp .env.example .env
```

Edit `.env`:
```
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
CARTESIA_API_KEY=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
```

Get your keys:
- Deepgram: https://console.deepgram.com
- OpenAI: https://platform.openai.com/api-keys
- Cartesia: https://play.cartesia.ai
- Twilio: https://console.twilio.com (Account SID + Auth Token on the dashboard)

## Run

**Terminal 1 — start the server:**
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — expose it publicly with ngrok:**
```bash
ngrok http 8000
```

**Twilio console — point your phone number's webhook to:**
```
https://<ngrok-subdomain>.ngrok.io/incoming-call
```
(Phone Numbers → Manage → Active Numbers → select your number → Voice webhook → set to the URL above)

Now call your Twilio number. The agent will answer.

## Project structure

```
app/
  main.py    # FastAPI server: /incoming-call webhook + /ws WebSocket pipeline
  agent.py   # SYSTEM_PROMPT (5 stages) + CallState dataclass
.env.example # API key template
pyproject.toml
```

## Key design decisions

**Twilio transport (`FastAPIWebsocketTransport` + `TwilioFrameSerializer`):**
Twilio streams audio as μ-law 8kHz JSON over a WebSocket. The serializer handles all encoding/decoding and automatically calls Twilio's REST API to hang up the call when the pipeline ends.

**`MuteUntilFirstBotCompleteUserMuteStrategy` (replaces `AlwaysUserMuteStrategy`):**
In v0, `AlwaysUserMuteStrategy` muted the mic any time the bot was speaking — necessary because the local microphone was picking up the speakers (echo). With Twilio, audio travels over the phone network so there is no echo. Instead, `MuteUntilFirstBotCompleteUserMuteStrategy` mutes the caller only during the opening greeting, then opens the line for full bidirectional conversation (including barge-in).

**VAD confidence=0.85:**
HVAC job sites have background noise: blowers, compressors, mechanical hum. Raising the Silero VAD confidence threshold from the default 0.7 to 0.85 forces the model to be 85% certain a sound is human speech before triggering the turn detector, filtering out most equipment noise.

## Tech stack

| Component | Service | Why |
|-----------|---------|-----|
| PSTN gateway | Twilio | Phone number + media streaming |
| STT | Deepgram Nova | Low-latency streaming transcription |
| LLM | GPT-4o-mini | Fast, cheap, reliable |
| TTS | Cartesia Sonic | Streaming TTS, <200ms first chunk |
| VAD | Silero (ONNX) | Runs locally, no extra API |
| Framework | Pipecat 1.5 | Audio pipeline + turn-taking |
| Server | FastAPI + uvicorn | Handles Twilio WebSocket + webhook |
