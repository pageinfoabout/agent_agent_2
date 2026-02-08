import requests
import io
import numpy as np

import requests
import numpy as np

class WhisperHTTPSTT:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()  # reuse connections

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # Safety: ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Clamp to [-1, 1] to avoid overflow
        audio = np.clip(audio, -1.0, 1.0)

        # Convert float32 → PCM16
        pcm16 = (audio * 32767.0).astype(np.int16)

        files = {
            "audio": (
                "audio.pcm",
                pcm16.tobytes(),
                "application/octet-stream",
            )
        }

        response = self.session.post(
            self.url,
            files=files,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        # Defensive check
        if "text" not in data:
            raise RuntimeError(f"Invalid STT response: {data}")

        return data["text"]
