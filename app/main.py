"""
HVAC voice agent — v1: Twilio PSTN gateway architecture.

Architecture:
    Caller (Phone) → Twilio (PSTN) ↔ Pipecat (Orchestrator)
                                       ├─ Deepgram (STT)
                                       ├─ GPT-4o-mini (LLM)
                                       └─ Cartesia (TTS)

Conversation flow (five stages, driven by SYSTEM_PROMPT):
    Greeting → Triage → Info Gathering → Confirmation → Goodbye

How it works:
    1. Twilio receives the inbound call and POSTs to /incoming-call.
    2. We return TwiML that tells Twilio to open a bidirectional media stream
       to wss://<your-host>/ws.
    3. The /ws WebSocket endpoint reads the Twilio "start" event to get the
       stream SID, then builds the Pipecat pipeline and runs it for the life
       of the call.
    4. TwilioFrameSerializer handles μ-law ↔ PCM conversion and auto-hangs up
       the Twilio call when the pipeline ends.

Run locally (requires ngrok or similar to get a public HTTPS URL):
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

Point Twilio's webhook to:
    https://<ngrok-subdomain>.ngrok.io/incoming-call
"""

import json
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)

from app.agent import SYSTEM_PROMPT

load_dotenv()

# Cartesia voice — swap the ID to change persona.
CARTESIA_VOICE_ID = "71a7ad14-091c-4e8e-a314-022ece01c121"

app = FastAPI()


def _require_key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(
            f"\n❌ Missing {name}.\n"
            f"   Add it to your .env file and restart.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def incoming_call(request: Request):
    """
    Twilio webhook — called when a phone call arrives.

    Returns TwiML that instructs Twilio to open a bidirectional WebSocket media
    stream to /ws on this server. Twilio then pipes the caller's audio in and
    plays back whatever audio we send out.
    """
    host = request.headers.get("host", request.url.hostname)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/ws" />
  </Connect>
</Response>"""
    return PlainTextResponse(twiml, media_type="text/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Twilio Media Streams WebSocket endpoint.

    Reads Twilio's initial "start" event to extract the stream and call SIDs,
    then builds the full Pipecat pipeline and runs it for the duration of the call.
    """
    await websocket.accept()

    deepgram_key = _require_key("DEEPGRAM_API_KEY")
    openai_key = _require_key("OPENAI_API_KEY")
    cartesia_key = _require_key("CARTESIA_API_KEY")
    twilio_account_sid = _require_key("TWILIO_ACCOUNT_SID")
    twilio_auth_token = _require_key("TWILIO_AUTH_TOKEN")

    # Consume Twilio's "connected" and "start" control events.
    # The WebSocket cursor advances past them, so the transport's receive loop
    # will start from the first "media" event — no messages are lost.
    stream_sid: str | None = None
    call_sid: str | None = None

    async for raw in websocket.iter_text():
        data = json.loads(raw)
        if data["event"] == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"]["callSid"]
            break

    if not stream_sid:
        await websocket.close()
        return

    # TwilioFrameSerializer:
    #   - deserialize: converts incoming μ-law 8kHz audio → PCM for Pipecat
    #   - serialize:   converts outgoing PCM audio → μ-law 8kHz for Twilio
    #   - auto_hang_up: POSTs to Twilio REST API to end the call when the
    #                   pipeline sends an EndFrame (i.e. Goodbye stage completes)
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=twilio_account_sid,
        auth_token=twilio_auth_token,
    )

    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
        ),
    )

    stt = DeepgramSTTService(api_key=deepgram_key)

    llm = OpenAILLMService(
        api_key=openai_key,
        settings=OpenAILLMService.Settings(model="gpt-4o-mini"),
    )

    tts = CartesiaTTSService(
        api_key=cartesia_key,
        settings=CartesiaTTSService.Settings(voice=CARTESIA_VOICE_ID),
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])

    # MuteUntilFirstBotCompleteUserMuteStrategy:
    #   Keeps the caller's audio muted until the bot finishes its Stage 1 greeting.
    #   After that the caller can speak — and interrupt — freely for the rest of
    #   the call. This replaces AlwaysUserMuteStrategy which was needed only for
    #   local audio to stop mic echo. Twilio is a phone call: the audio path is
    #   Caller → Twilio → WebSocket → server, so there is no echo to worry about.
    #
    # VADParams confidence=0.85:
    #   Requires 85% model certainty before treating a sound as human speech.
    #   This rejects HVAC mechanical hum, line noise, and background fan noise
    #   that a looser threshold would misread as the caller speaking.
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.85,
                    start_secs=0.3,
                    stop_secs=0.8,
                    min_volume=0.6,
                )
            ),
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),        # audio in from Twilio
            stt,                      # Deepgram: audio → transcript
            aggregators.user(),       # VAD + turn detection + context accumulation
            llm,                      # GPT-4o-mini: transcript → response text
            tts,                      # Cartesia: response text → audio
            transport.output(),       # audio out to Twilio
            aggregators.assistant(),  # record what the bot said
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket):
        """Cancel the pipeline when the caller hangs up."""
        await task.cancel()

    # Trigger the Stage 1 greeting immediately when the call connects.
    await task.queue_frames([LLMRunFrame()])

    await runner.run(task)
