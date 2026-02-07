from livekit.agents import tts
import livekit.agents.utils as utils
import io
import aiohttp
import logging
import asyncio
from typing import Optional
from pathlib import Path
import numpy as np
import os
os.environ["TORCH_BACKEND"] = "none"
import torch

logger = logging.getLogger("cosyvoice")

class CosyVoiceTTS(tts.TTS):
    def __init__(self, base_url: str, prompt_text: str, prompt_wav_path: str, sample_rate: int = 16000):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),  # CosyVoice не streaming
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._base_url = f"{base_url.rstrip('/')}/inference_zero_shot"
        self._prompt_text = prompt_text
        self._prompt_wav_path = str(Path(prompt_wav_path).resolve())
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"CosyVoiceTTS: {self._base_url}")

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=30)
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    # ... твой _get_session() без изменений ...

    def synthesize(self, text: str, *, conn_options=None) -> "CosyVoiceChunkedStream":
        return CosyVoiceChunkedStream(
           tts=self,
        input_text=text,
        conn_options=conn_options
        )

class CosyVoiceChunkedStream(tts.ChunkedStream):
    # Твой __init__ без изменений
    

    async def _run(self, output_emitter: tts.AudioEmitter):
        sample_rate = self._tts.sample_rate
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=sample_rate,
            num_channels=1,
            mime_type="audio/pcm"  # LiveKit ожидает PCM
        )

        try:
            if not Path(self._prompt_wav_path).exists():
                logger.error(f"Prompt WAV missing: {self._prompt_wav_path}")
                return

            # 🎯 ТОЧНО как в твоем Python клиенте!
            form = aiohttp.FormData()
            form.add_field('tts_text', self._input_text)           # От LLM
            form.add_field('prompt_text', self._prompt_text)       # Фиксированный
            
            with open(self._prompt_wav_path, 'rb') as f:
                form.add_field(
                    'rr', 
                    f, 
                    filename='rr',                    # ← Точно!
                    content_type='application/octet-stream'   # ← Точно!
                )

            logger.info(f"CosyVoice → '{self._input_text[:50]}...'")

            async with self._session.post(self._url, data=form) as resp:
                if resp.status != 200:
                    logger.error(f"HTTP {resp.status}: {await resp.text()}")
                    return

                # CosyVoice возвращает WAV → декодируем → PCM чанки
                full_audio = b""
                async for chunk in resp.content.iter_chunked(8192):
                    full_audio += chunk

                # 🔥 WAV → PCM 16-bit (используем torchaudio как в сервере)
                import torchaudio
                speech, sr = torchaudio.load(io.BytesIO(full_audio))
                
                # Resample если нужно (CosyVoice 22kHz → LiveKit 16/24kHz)
                if sr != self._sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self._sample_rate)
                    speech = resampler(speech)
                
                pcm_bytes = (speech.squeeze(0).numpy() * 32767).astype(np.int16).tobytes()

                # 20ms чанки = 320 bytes @ 16kHz
                chunk_size = int(self._sample_rate * 0.02 * 2)  # 16-bit
                for i in range(0, len(pcm_bytes), chunk_size):
                    audio_chunk = pcm_bytes[i:i+chunk_size]
                    if audio_chunk:
                        output_emitter.push(audio_chunk)

        except Exception as e:
            logger.error(f"CosyVoice error: {e}")
        
        output_emitter.flush()
