from egtts import Egtts
# from voice_agent.whisper_stt import LocalWhisperSTT
from livekit.agents.llm import LLM
import os
from openai import OpenAI
from livekit.agents.llm import ChatContext
from contextlib import asynccontextmanager

class OpenAILLM(LLM):
    def __init__(self):
        super().__init__()
        self.client = OpenAI()
        self.history = [
            {"role": "system", "content": "You are a helpful Egyptian voice assistant."}
        ]

    @asynccontextmanager
    async def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools=None,
        conn_options=None,
        parallel_tool_calls=None,
        tool_choice=None,
        extra_kwargs=None,
    ):
        prompt = chat_ctx.latest_user_message()
        self.history.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.history
        )

        message = completion.choices[0].message.content
        try:
            yield message 
        finally:
            pass

from livekit.agents.llm import LLM

openai_api_key = os.getenv("OPENAI_API_KEY")

class OpenRouterLLM(LLM):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_api_key
        )
        self.history = [
            {"role": "system", "content": "You are a helpful voice assistant."}
        ]

    def chat(self, prompt: str,chat_ctx=None, tools=None, tool_choice=None) -> str:
        self.history.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(
          model="deepseek/deepseek-chat-v3-0324:free",
          messages=self.history
        )
        return completion.choices[0].message.content


import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
)
from livekit.plugins import (
    noise_cancellation,
    silero,
    cartesia,
    groq,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()
logger = logging.getLogger("voice-agent")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant. That speeks in Egyptian Arabic.")

def prewarm(proc: JobProcess):
    """Prewarm function - load models in each worker process"""
    logger.info("Prewarm: Loading models in worker process...")

    try:
        logger.info("Loading VAD model...")
        proc.userdata["vad"] = silero.VAD.load()

        logger.info("Loading TTS model...")
        proc.userdata["tts_model"] = Egtts()

        logger.info("Prewarm completed - all models loaded in worker process!")

    except Exception as e:
        logger.error(f"Error loading models in prewarm: {e}")
        raise

from livekit import agents
from livekit.plugins import openai

async def entrypoint(ctx: agents.JobContext):
    """Entrypoint - use preloaded models from worker process"""
    logger.info("Starting entrypoint with preloaded models...")

    vad_model = ctx.proc.userdata["vad"]
    # whisper_model = ctx.proc.userdata["whisper_model"]
    # seamless = ctx.proc.userdata["seamless"]
    tts_model = ctx.proc.userdata["tts_model"]

    logger.info("Loading turn detection model...")
    turn_detector = MultilingualModel()

    session = AgentSession(
        # stt=openai.STT(model="gpt-4o-mini-transcribe", language="ar"),
        stt = cartesia.STT(model="ink-whisper"),# model = "ink-whisper-2025-06-04",
        # llm=openai.LLM(model="gpt-4o-mini"),
        llm=groq.LLM(model="llama3-8b-8192"),

        tts=tts_model,
        vad=vad_model,
        turn_detection=turn_detector,
        max_endpointing_delay=30,
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance. In Egyptian Arabic"
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,
        ),
    )
