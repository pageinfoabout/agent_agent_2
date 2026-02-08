import asyncio
import time
import numpy as np
import requests
from types import SimpleNamespace
from livekit.agents.stt import stt  # Make sure LiveKit stt base classes are imported

class WhisperHTTPStream(stt.RecognizeStream):
    """
    Stream-like wrapper for HTTP Whisper batch server.
    Mimics the behavior of streaming STT for LiveKit.
    """

    def __init__(self, url: str, sample_rate: int = 16000):
        super().__init__(stt=None, sample_rate=sample_rate)
        self.url = url
        self.session = requests.Session()

        # LiveKit expects capabilities and event support
        self.capabilities = SimpleNamespace(streaming=True)
        self._event_ch = asyncio.Queue()  # Used by _emit_final / _emit_interim
        self._last_interim = ""
        self._last_interim_time = 0.0

    # Fake `.on` to satisfy StreamAdapter
    def on(self, event_name, callback):
        # HTTP STT does not generate metrics events
        pass

    async def run_stream(self, audio_chunks: asyncio.Queue):
        """
        Accepts audio chunks from LiveKit (VADed segments)
        Each chunk is a numpy float32 array, sends to batch HTTP Whisper,
        emits interim + final events.
        """
        while True:
            chunk = await audio_chunks.get()
            if chunk is None:  # End of stream
                # Emit final if last interim exists
                if self._last_interim:
                    self._emit_final(self._last_interim)
                break

            # Convert chunk to PCM16 bytes
            audio_bytes = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()

            # Send to Flask Whisper server
            try:
                response = self.session.post(
                    self.url,
                    files={"audio": ("audio.pcm", audio_bytes, "application/octet-stream")},
                    timeout=30
                )
                response.raise_for_status()
                text = response.json().get("text", "").strip()
            except Exception as e:
                print(f"[STT] HTTP error: {e}")
                continue

            # Emit interim event
            if text and text != self._last_interim:
                self._last_interim = text
                self._last_interim_time = time.time()
                self._emit_interim(text)

            # Optional: short delay to simulate streaming chunks
            await asyncio.sleep(0.05)

            # Emit final after each chunk (or could batch multiple chunks)
            if text:
                self._emit_final(text)

    def _emit_final(self, text: str):
        try:
            event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="ru", text=text)],
            )
            self._event_ch.put_nowait(event)
            self._last_interim = ""
        except Exception as e:
            print(f"[STT] Failed to emit final: {e}")

    def _emit_interim(self, text: str):
        try:
            event = stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="ru", text=text)],
            )
            self._event_ch.put_nowait(event)
        except Exception as e:
            print(f"[STT] Failed to emit interim: {e}")
