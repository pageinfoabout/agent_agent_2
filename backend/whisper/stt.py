import requests
import io
import numpy as np

class WhisperHTTPSTT:
    def __init__(self, url: str):
        self.url = url

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # Convert float32 → PCM16
        pcm16 = (audio * 32767).astype(np.int16)

        files = {
            "audio": ("audio.pcm", pcm16.tobytes(), "application/octet-stream")
        }

        r = requests.post(self.url, files=files, timeout=30)
        r.raise_for_status()
        return r.json()["text"]
