import whisper
from livekit.agents.stt import STT


class WhisperSTT(STT):
    def __init__(self):
        self.model = whisper.load_model("turbo", device = "cuda")

    async def transcribe(self, audio: bytes, sample_rate: int) -> str:
        # Convert audio to numpy/wav if needed
        result = self.model.transcribe(audio)
        return result['text']