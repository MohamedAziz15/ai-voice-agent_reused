import time
from livekit.agents.stt import STT, STTCapabilities, SpeechEvent, SpeechEventType
from faster_whisper import WhisperModel
import io
import numpy as np
import wave

class SpeechAlternative:
    """Simple class to hold speech recognition alternatives."""
    def __init__(self, text: str, confidence: float):
        self.text = text
        self.confidence = confidence

class LocalWhisperSTT(STT):
    def __init__(self, model=None):
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=True))
        # Use preloaded model if provided, otherwise load it
        self.model = model if model is not None else WhisperModel("large", compute_type="int8", device="cuda")

    async def _recognize_impl(self, buffer: bytes, *, language: str = None, conn_options=None) -> SpeechEvent:
        try:
            # Convert audio bytes to temporary file for Whisper
            with io.BytesIO() as audio_buffer:
                # Write WAV header for the audio data
                with wave.open(audio_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz sample rate
                    wav_file.writeframes(buffer)

                audio_buffer.seek(0)

                # Transcribe with Whisper
                segments, info = self.model.transcribe(
                    audio_buffer,
                    language="ar",  # Default to Arabic
                    beam_size=5,
                    word_timestamps=True
                )

                # Combine all segments
                text = " ".join(segment.text.strip() for segment in segments)

                # Return final speech event with proper alternative objects
                return SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechAlternative(text=text, confidence=0.9)]
                )

        except Exception as e:
            print(f"STT Error: {e}")
            # Return empty result on error
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechAlternative(text="", confidence=0.0)]
            )


if __name__ == "__main__":
    model = LocalWhisperSTT()
    print(model.model)
    time.sleep(30)