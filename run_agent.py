from egtts import Egtts
from stt import LocalWhisperSTT
from livekit.agents.llm import LLM
import google.generativeai as genai
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
        # Extract prompt from chat_ctx
        prompt = chat_ctx.latest_user_message()
        self.history.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model="gpt-4o",  # or whatever is valid
            messages=self.history
        )

        message = completion.choices[0].message.content
        try:
            yield message  # simulate a stream of one response
        finally:
            pass

from livekit.agents.llm import LLM

class OpenRouterLLM(LLM):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
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

class GeminiLLM(LLM):
    def __init__(self):
        super().__init__()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')

    async def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    async def chat(self, messages: list[dict], chat_ctx=None, tools=None, tool_choice=None) -> str:
        formatted_messages = [
            {"role": msg["role"], "parts": [msg["content"]]}
            for msg in messages
        ]
        response = self.model.generate_content(formatted_messages)
        return response.text


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
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()
logger = logging.getLogger("voice-agent")

# Global model storage - loaded once at startup
GLOBAL_MODELS = {}

def load_models():
    """Load models that can be loaded without job context"""
    logger.info("Loading models globally...")

    try:
        # Load VAD model
        logger.info("Loading VAD model...")
        GLOBAL_MODELS["vad"] = silero.VAD.load()

        # Load Whisper STT model
        logger.info("Loading Whisper STT model...")
        GLOBAL_MODELS["whisper"] = LocalWhisperSTT()

        # Load TTS model
        logger.info("Loading TTS model...")
        GLOBAL_MODELS["tts"] = Egtts()

        # Note: Turn detection model cannot be loaded here as it requires job context
        # It will be loaded in entrypoint() where job context is available

        logger.info("Global models loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant. That speeks in Egyptian Arabic.")

def prewarm(proc: JobProcess):
    """Prewarm function - most models are already loaded globally"""
    logger.info("Prewarm: Setting references to preloaded models...")

    # Set references to global models in userdata for backward compatibility
    proc.userdata["vad"] = GLOBAL_MODELS["vad"]
    proc.userdata["whisper_model"] = GLOBAL_MODELS["whisper"]
    proc.userdata["tts_model"] = GLOBAL_MODELS["tts"]

    logger.info("Prewarm completed - model references set!")

from livekit import agents
from livekit.plugins import openai

async def entrypoint(ctx: agents.JobContext):
    """Entrypoint - use preloaded models and load turn detector here"""
    logger.info("Starting entrypoint with preloaded models...")

    # Use preloaded models from global storage
    vad_model = GLOBAL_MODELS["vad"]
    whisper_model = GLOBAL_MODELS["whisper"]
    tts_model = GLOBAL_MODELS["tts"]

    # Load turn detection model here where job context is available
    logger.info("Loading turn detection model...")
    turn_detector = MultilingualModel()

    session = AgentSession(
        
        stt=LocalWhisperSTT(model=whisper_model),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=tts_model,
        vad=vad_model,
        turn_detection=turn_detector,
        max_endpointing_delay=30,
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance. In Egyptian Arabic"
    )

if __name__ == "__main__":
    import sys

    sys.argv = ['run_agent.py', 'connect', '--url', 'wss://ai-project-zu8ypdw6.livekit.cloud', '--api-key', 'APIfX2YSQJkg4Hz', '--api-secret', '2f9DtZlkm6gZ2cY2firv3BTsDU2o0KMf8inLTK6eclzB', '--room', 'playground-EyWj-0prN']

    # Load all models once at startup
    load_models()

    # Run the agent with preloaded models
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )