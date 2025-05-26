import time
import logging
from livekit.agents.stt import STT, STTCapabilities, SpeechEvent, SpeechEventType
from faster_whisper import WhisperModel
import io
import numpy as np
import wave

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SpeechAlternative:
    """Simple class to hold speech recognition alternatives."""
    def __init__(self, text: str, confidence: float):
        self.text = text
        self.confidence = confidence


class LocalWhisperSTT(STT):
    def __init__(self, model=None):
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        # Use preloaded model if provided, otherwise load it
        self.model = model if model is not None else WhisperModel("large", compute_type="float16", device="cuda")
        logger.info(f"Whisper model loaded: {self.model}")
    
    async def _recognize_impl(self, buffer, *, language: str = None, conn_options=None) -> SpeechEvent:
        try:
            # Handle AudioFrame object from LiveKit
            if hasattr(buffer, 'data'):
                # It's an AudioFrame, extract the raw audio data
                audio_data = buffer.data
                sample_rate = buffer.sample_rate
                channels = buffer.num_channels
                logger.debug(f"Received AudioFrame: sample_rate={sample_rate}, channels={channels}, data_len={len(audio_data)}")
            else:
                # It's raw bytes
                audio_data = buffer
                sample_rate = 16000  # Assume default
                channels = 1
                logger.debug(f"Received raw audio buffer of {len(audio_data)} bytes")
            
            # Check if buffer is empty or too small
            if not audio_data or len(audio_data) < 1000:  # Less than ~0.03 seconds at 16kHz
                logger.warning(f"Buffer too small: {len(audio_data)} bytes")
                return SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechAlternative(text="", confidence=0.0)]
                )
            
            # Convert audio bytes to numpy array first for better handling
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            logger.debug(f"Audio array shape: {audio_array.shape}, max: {audio_array.max():.4f}, min: {audio_array.min():.4f}")
            
            # Handle multi-channel audio by taking only the first channel
            if channels > 1:
                # Reshape to (samples, channels) and take first channel
                audio_array = audio_array.reshape(-1, channels)[:, 0]
                logger.debug(f"Converted from {channels} channels to mono")
            
            # Check if audio has actual content (not just silence)
            audio_rms = np.sqrt(np.mean(audio_array**2))
            logger.debug(f"Audio RMS level: {audio_rms:.6f}")
            
            if audio_rms < 0.001:  # Very quiet audio
                logger.warning("Audio appears to be silence")
                return SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechAlternative(text="", confidence=0.0)]
                )
            
            # Resample if necessary (Whisper expects 16kHz)
            if sample_rate != 16000:
                logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz")
                # Simple resampling (you might want to use librosa for better quality)
                duration = len(audio_array) / sample_rate
                new_length = int(duration * 16000)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), new_length),
                    np.arange(len(audio_array)),
                    audio_array
                )
            
            # Convert back to 16-bit for WAV format
            audio_int16 = (audio_array * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Create proper WAV file in memory
            with io.BytesIO() as audio_buffer:
                with wave.open(audio_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz sample rate
                    wav_file.writeframes(audio_bytes)
                
                audio_buffer.seek(0)
                logger.debug(f"WAV buffer size: {len(audio_buffer.getvalue())} bytes")
                
                # Transcribe with Whisper
                logger.debug("Starting Whisper transcription...")
                segments, info = self.model.transcribe(
                    audio_buffer,
                    language=language or "ar",  # Use provided language or default to Arabic
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True,  # Voice activity detection
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                logger.debug(f"Whisper info: {info}")
                
                # Combine all segments
                segments_list = list(segments)
                logger.debug(f"Number of segments: {len(segments_list)}")
                
                if not segments_list:
                    logger.warning("No segments detected by Whisper")
                    return SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[SpeechAlternative(text="", confidence=0.0)]
                    )
                
                text = " ".join(segment.text.strip() for segment in segments_list)
                logger.info(f"Transcribed text: '{text}'")
                
                # Calculate average confidence if available
                avg_confidence = sum(getattr(segment, 'avg_logprob', -1.0) for segment in segments_list) / len(segments_list)
                # Convert log probability to confidence (rough approximation)
                confidence = max(0.0, min(1.0, np.exp(avg_confidence))) if avg_confidence > -10 else 0.5
                
                # Return final speech event with proper alternative objects
                return SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechAlternative(text=text, confidence=confidence)]
                )
                
        except Exception as e:
            logger.error(f"STT Error: {e}", exc_info=True)
            # Return empty result on error
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechAlternative(text="", confidence=0.0)]
            )

    def test_with_sample_audio(self, sample_rate=16000, duration=3.0, frequency=440):
        """Generate a test tone to verify the STT pipeline works"""
        logger.info("Testing STT with generated tone...")
        
        # Generate a sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Test the recognition
        import asyncio
        result = asyncio.run(self._recognize_impl(audio_bytes))
        logger.info(f"Test result: {result}")
        return result

if __name__ == "__main__":
    # Test the STT implementation
    stt = LocalWhisperSTT()
    
    # Test with generated audio
    stt.test_with_sample_audio()
    
    print("STT instance created successfully")
    time.sleep(5)