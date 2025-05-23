import torch
import torchaudio
from livekit.agents.tts import TTS
from huggingface_hub import snapshot_download
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig


import io
import os
import logging


class Egtts(TTS):
    def __init__(self):
        device = "cuda"
        repo_id = "OmarSamir/EGTTS-V0.1"
        logging.info("⏳Downloading EGTTS model.")
        
        os.makedirs("./models_path", exist_ok=True)
        curr_dir_path = os.path.dirname(os.path.realpath(__file__))
        local_dir = os.path.join(curr_dir_path, "models_path")
        model_path = snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)


        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        # self.model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        self.model.load_checkpoint(
            config, checkpoint_dir=model_path, eval=True, use_deepspeed=False
        )
        self.model.to(device)
        # self.compiled_model = torch.compile(self.model.inference_stream)

        speaker_audio_path = os.path.join(model_path, "speaker_reference.wav")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[speaker_audio_path])
        
        self.speaker_embedding = speaker_embedding
        self.gpt_cond_latent = gpt_cond_latent
        logging.info("🔥Model Loaded")

    async def synthesize(self, text: str) -> bytes:
        print(f"🗣 Synthesizing: {text}")
        outputs = self.model.inference(
            text=text,
            language=self.language,
            gpt_cond_latent=self.gpt_cond_latent,
            speaker_embedding=self.speaker_embedding
        )

        waveform = outputs["wav"].unsqueeze(0)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform.cpu(), sample_rate=16000, format="wav")
        wav_bytes = buffer.getvalue()

        # Return raw PCM (without WAV header)
        return wav_bytes[44:]

