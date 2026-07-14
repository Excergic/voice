"""
HVAC voice agent — STAGE 2: full spoken conversation loop.

mic -> STT -> LLM -> TTS -> speakers, with barge-in.

Now it TALKS BACK. The agent greets you, you answer, it responds out loud.
You can interrupt it mid-sentence and it stops (barge-in). There are still no
tools, no call state, and no safety guard yet — that's stage 4.

Pipeline order (this order matters):
    mic in -> STT -> user_aggregator -> LLM -> TTS -> speakers -> assistant_aggregator

The two "aggregators" are Pipecat's memory: the user one collects your speech
into the conversation history (and, crucially, holds the VAD that powers
turn-taking and barge-in); the assistant one records what the bot said.

Run it:   uv run python -m app.main
Stop it:  Ctrl+C
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

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
from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from app.agent import SYSTEM_PROMPT

load_dotenv()

# A neutral, clear Cartesia voice (their documented default). Swap the voice_id
# later if you want a different persona.
CARTESIA_VOICE_ID = "71a7ad14-091c-4e8e-a314-022ece01c121"


def _require_key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(
            f"\n❌ Missing {name}.\n"
            f"   1. Copy the template:  cp .env.example .env\n"
            f"   2. Paste your key after {name}= in .env (no quotes, no spaces).\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


async def main():
    deepgram_key = _require_key("DEEPGRAM_API_KEY")
    openai_key = _require_key("OPENAI_API_KEY")
    cartesia_key = _require_key("CARTESIA_API_KEY")

    # Mic AND speakers now — audio flows both directions.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = DeepgramSTTService(api_key=deepgram_key)

    # gpt-4o-mini: cheapest tool-calling OpenAI model, low latency. That latency
    # bias is exactly what we want for a spoken dispatcher.
    llm = OpenAILLMService(
        api_key=openai_key,
        settings=OpenAILLMService.Settings(model="gpt-4o-mini"),
    )

    # Cartesia streaming TTS. Defaults to the sonic-3.5 model.
    tts = CartesiaTTSService(
        api_key=cartesia_key,
        settings=CartesiaTTSService.Settings(voice=CARTESIA_VOICE_ID),
    )

    # The shared conversation memory, seeded with the system prompt.
    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])

    # The aggregator PAIR. The user side gets the Silero VAD analyzer — in
    # Pipecat 1.5 this is where turn-taking and barge-in live (not the transport).
    #
    # AlwaysUserMuteStrategy: mutes mic input while the bot is speaking.
    # This prevents the microphone from picking up the speakers (echo), which
    # was causing the VAD to fire false "user started speaking" interruptions
    # every ~600ms and cutting the bot off mid-sentence.
    #
    # VADParams: stop_secs=0.8 gives the user 800ms of silence before the turn
    # is committed (the default 0.2s is far too aggressive for natural speech).
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.3,
                    stop_secs=0.8,
                    min_volume=0.6,
                )
            ),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    print("\n📞 Call connected. The dispatcher will greet you — then just talk.")
    print("   Try interrupting it mid-sentence; it should stop (barge-in).")
    print("   Press Ctrl+C to hang up.\n")

    # Make the bot speak first: queue a run so it generates a greeting from the
    # system prompt before you say anything.
    await task.queue_frames([LLMRunFrame()])

    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Call ended.")
