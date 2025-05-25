import os
import torch
import torchaudio
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from livekit.agents.tts import TTS, TTSCapabilities, SynthesizedAudio
from livekit.rtc import AudioFrame
from huggingface_hub import snapshot_download

import io
import asyncio



class Egtts(TTS):
    def __init__(self):
        super().__init__(capabilities=TTSCapabilities(streaming=False), sample_rate=22050, num_channels=1)
        snapshot_path = snapshot_download(repo_id="OmarSamir/EGTTS-V0.1")
        config_path = os.path.join(snapshot_path, "config.json")
        vocab_path = os.path.join(snapshot_path, "vocab.json")
        speaker_audio_path = os.path.join(snapshot_path, "speaker_reference.wav")

        self.language = "ar"

        print("🔊 Loading XTTS model...")
        config = XttsConfig()
        config.load_json(config_path)

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=snapshot_path, use_deepspeed=True, vocab_path=vocab_path)
        self.model.cuda()
        self.model.eval()

        print("🎤 Computing speaker latents...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[speaker_audio_path]
        )

    def _wav_postprocess(self, wav):
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    async def synthesize(self, text: str, *, conn_options=None, **kwargs):
        chunks = self.model.inference_stream(
                text,
                self.language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                stream_chunk_size=20,
                enable_text_splitting=True,
            )
        
        # Convert chunks to list to determine final chunk
        chunk_list = list(chunks)
        total_chunks = len(chunk_list)
        
        for i, chunk in enumerate(chunk_list):
            is_final = (i == total_chunks - 1)  # Last chunk is final
            
            print(f"Processing chunk {i}/{total_chunks}, is_final: {is_final}, shape: {chunk.shape if hasattr(chunk, 'shape') else 'unknown'}")
            
            # Process the audio chunk
            processed_chunk = self._wav_postprocess(chunk)
            print(f"Processed chunk {i}, length: {len(processed_chunk)}")
            
            # Create LiveKit audio frame
            audio_frame = AudioFrame(
                data=processed_chunk.tobytes(),
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
                samples_per_channel=len(processed_chunk)
            )
            
            print(f"Created audio frame {i}")
            
            # Yield as SynthesizedAudio event with request_id and is_final
            yield SynthesizedAudio(
                frame=audio_frame,
                request_id=kwargs.get('request_id', None),
                is_final=is_final
            )