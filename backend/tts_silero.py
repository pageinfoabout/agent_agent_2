from livekit.agents import tts
import numpy as np
import os
os.environ["TORCH_BACKEND"] = "none"
import torch
from typing import Optional
import logging

logger = logging.getLogger("silero_tts")


class LocalSileroTTS(tts.TTS):
    """Local Silero TTS using v5_ru model."""
    
    def __init__(
        self,
        language: str = "ru",
        model_id: str = "v5_ru",
        speaker: str = "baya",
        device: str = "cpu",
        sample_rate: int = 48000,
        put_accent: bool = True,
        put_yo: bool = True,
        put_stress_homo: bool = False,
        put_yo_homo: bool = True,
        
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self.language = language
        self.model_id = model_id
        self.speaker = speaker
        self.device = torch.device(device)
        self.put_accent = put_accent
        self.put_yo = put_yo
        self.put_stress_homo = put_stress_homo
        self.put_yo_homo = put_yo_homo
        self._model = None
        self._example_text = None
    
    def _load_model(self):
        """Lazy load the Silero TTS model."""
        if self._model is None:
            try:
                logger.info(f"Loading Silero model: {self.model_id} (speaker: {self.speaker})")
                self._model, self._example_text = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=self.language,
                    speaker=self.model_id,
                    trust_repo=True
                )
                self._model.to(self.device)
                # Silero models are already in eval mode, no need to call .eval()
                logger.info(f"Silero model loaded successfully. Available speakers: {self._model.speakers}")
            except Exception as e:
                logger.error(f"Failed to load Silero model: {e}", exc_info=True)
                raise
        return self._model
    
    def synthesize(self, text: str, *, conn_options=None) -> "LocalSileroTTSStream":
        return LocalSileroTTSStream(
            tts=self,
            text=text,
            conn_options=conn_options,
        )


class LocalSileroTTSStream(tts.ChunkedStream):
    """Stream for local Silero TTS synthesis."""
    
    def __init__(self, tts: LocalSileroTTS, text: str, conn_options=None):
        super().__init__(
            tts=tts,
            input_text=text,
            conn_options=conn_options,
        )
    
    async def _run(self, output_emitter: tts.AudioEmitter):
        """Generate audio from text using local Silero model."""
        # Initialize emitter
        output_emitter.initialize(
            request_id="silero-tts",
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        
        # Load model if not already loaded
        model = self._tts._load_model()
        
        # Generate audio using the same parameters as tts_silero.py
        def generate_audio():
            audio = model.apply_tts(
                text=self._input_text,
                speaker=self._tts.speaker,
                sample_rate=self._tts.sample_rate,
                put_accent=self._tts.put_accent,
                put_yo=self._tts.put_yo,
                put_stress_homo=self._tts.put_stress_homo,
                put_yo_homo=self._tts.put_yo_homo,
                
            )
            return audio
        
        # Run synthesis in executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, generate_audio)
        
        # Convert to numpy array if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        elif not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # Convert float32 [-1, 1] to int16 PCM
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Emit audio in chunks (4096 bytes = ~42ms at 48kHz 16-bit mono)
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            output_emitter.push(chunk)
        
        output_emitter.flush()