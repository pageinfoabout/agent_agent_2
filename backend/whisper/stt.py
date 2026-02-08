# stt_adapter.py
import numpy as np
import requests
import asyncio

class WhisperHTTPSTT:
    def __init__(self, url: str, buffer_size: int = 16000):
        """
        url: Flask server URL
        buffer_size: number of samples to batch (~1s at 16kHz)
        """
        self.url = url
        self.buffer_size = buffer_size
        self.buffer = np.array([], dtype=np.float32)

    def add_chunk(self, audio_chunk: np.ndarray):
        """Append audio chunk to buffer"""
        self.buffer = np.concatenate([self.buffer, audio_chunk])

    async def transcribe_buffer(self):
        """Send current buffer to Flask server"""
        if len(self.buffer) == 0:
            return ""

        # Convert float32 back to int16 for Flask server
        audio_int16 = (self.buffer * 32768.0).astype(np.int16).tobytes()
        files = {'audio': ('chunk.wav', audio_int16, 'audio/wav')}
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(self.url, files=files))
        self.buffer = np.array([], dtype=np.float32)  # clear buffer
        return response.json().get("text", "")

    async def process_chunk(self, audio_chunk: np.ndarray):
        """Add chunk and transcribe if buffer is full"""
        self.add_chunk(audio_chunk)
        if len(self.buffer) >= self.buffer_size:
            text = await self.transcribe_buffer()
            if text:
                print("Transcribed:", text)
