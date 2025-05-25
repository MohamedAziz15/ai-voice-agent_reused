import os
import torch
import torchaudio
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from livekit.plugins.xtts import TTS, Voice
from huggingface_hub import snapshot_download

import io
import asyncio



class Egtts:
    def __init__(self):
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
        yield wav

    def synthesize(self, text: str, *, conn_options=None, **kwargs):
        # Simulate streaming audio in 3 chunks
        chunks = self.model.inference_stream(
            text,
            self.language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            stream_chunk_size=20,
            enable_text_splitting=True,
        )
        for chunk in chunks:
            processed_chunk = self._wav_postprocess(chunk)
        #     processed_bytes = processed_chunk.tobytes()
        #     yield processed_bytes
        # output = self.model.inference(
        #     text,
        #     self.language,
        #     self.gpt_cond_latent,
        #     self.speaker_embedding,
        # )
        # yield self._wav_postprocess(output).tobytes()
        return chunks


if __name__ == "__main__":
    egtts = Egtts()
    output = egtts.synthesize("صباح الخير")
    print(output)
    
