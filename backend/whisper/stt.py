import asyncio
import dataclasses
import numpy as np
from dataclasses import dataclass
from typing import Any

from transformers import pipeline
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr


# ---------------- OPTIONS ----------------

@dataclass
class STTOptions:
    language: str = "en"
    sample_rate: int = 16000


# ---------------- STT ----------------

class WhisperSTT(stt.STT):
    def __init__(self, pipe: pipeline):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=False,
            )
        )
        self.pipe = pipe
        self._opts = STTOptions()

    @property
    def provider(self) -> str:
        return "whisper"

    @property
    def model(self) -> str:
        return "local-whisper"

    def stream(self, *, language: NotGivenOr[str] = NOT_GIVEN, conn_options: Any = None):
        opts = dataclasses.replace(self._opts)
        if language is not NOT_GIVEN:
            opts.language = language

        return WhisperSpeechStream(stt=self, opts=opts)


# ---------------- STREAM ----------------

class WhisperSpeechStream(stt.SpeechStream):
    class _Flush:
        pass

    def __init__(self, *, stt: WhisperSTT, opts: STTOptions):
        super().__init__(stt=stt, conn_options=None, sample_rate=opts.sample_rate)
        self.opts = opts
        self.queue = asyncio.Queue()
        self.samples: list[np.ndarray] = []

    async def _run(self):
        while True:
            frame = await self.queue.get()

            if isinstance(frame, self._Flush):
                if self.samples:
                    await self._transcribe()
                break

            pcm = np.frombuffer(frame.data, dtype=np.int16)
            self.samples.append(pcm)

            # transcribe once we have ~1s audio
            if sum(len(x) for x in self.samples) >= self.opts.sample_rate:
                await self._transcribe()

    async def _transcribe(self):
        audio = np.concatenate(self.samples)
        self.samples.clear()

        result = self.stt.pipe(
            audio.astype(np.float32) / 32768.0,
            sampling_rate=self.opts.sample_rate,
        )

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id="whisper",
                alternatives=[
                    stt.SpeechData(
                        text=result["text"],
                        language=self.opts.language,
                        confidence=1.0,
                        start_time=0,
                        end_time=0,
                        words=None,
                    )
                ],
            )
        )

    async def aclose(self):
        await self.queue.put(self._Flush())
        await super().aclose()
