# whisper_stt.py
from __future__ import annotations
import asyncio
import dataclasses
import weakref
from dataclasses import dataclass
from typing import Any, Sequence, Optional

import torch
from transformers import pipeline
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NotGivenOr, NOT_GIVEN
from livekit.agents.voice.io import TimedString
from livekit.agents.utils import AudioBuffer

@dataclass
class STTOptions:
    language: str
    sample_rate: int
    interim_results: bool = True
    chunk_length_s: float = 10.0  # chunk length for streaming

class WhisperSTT(stt.STT):
    def __init__(
        self,
        pipe: pipeline,
        language: str = "en",
        sample_rate: int = 16000,
        chunk_length_s: float = 10.0,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False,
                aligned_transcript="word",
            )
        )
        self.pipe = pipe
        self._opts = STTOptions(
            language=language,
            sample_rate=sample_rate,
            chunk_length_s=chunk_length_s,
        )
        self._streams = weakref.WeakSet()

    @property
    def provider(self) -> str:
        return "Whisper"

    @property
    def model(self) -> str:
        return str(self.pipe.model.config._name_or_path)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: Any = None,
    ) -> stt.SpeechEvent:
        # Merge options
        lang = self._opts.language if not is_given(language) else language
        audio_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(audio_bytes)
            f.flush()
            result = self.pipe(f.name)

        # Convert to STT event
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id="local_whisper",
            alternatives=[stt.SpeechData(
                language=lang,
                start_time=0,
                end_time=0,
                confidence=result.get("score", 1.0),
                text=result["text"],
                words=None,  # Whisper does not return per-word timing
            )]
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: Any = None,
    ) -> WhisperSpeechStream:
        opts = dataclasses.replace(self._opts)
        if is_given(language):
            opts.language = language
        stream = WhisperSpeechStream(stt=self, opts=opts)
        self._streams.add(stream)
        return stream

def is_given(val) -> bool:
    return val is not NOT_GIVEN

class WhisperSpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: WhisperSTT, opts: STTOptions):
        super().__init__(stt=stt, conn_options=None, sample_rate=opts.sample_rate)
        self._opts = opts
        self._speaking = False
        self._input_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()

    async def _run(self) -> None:
        buffer = AudioBuffer(sample_rate=self._opts.sample_rate)
        chunk_frames = int(self._opts.chunk_length_s * self._opts.sample_rate)

        while True:
            data = await self._input_queue.get()
            if isinstance(data, self._FlushSentinel):
                # Process remaining audio
                if buffer.num_frames > 0:
                    await self._process_chunk(buffer)
                    buffer.clear()
                break

            buffer.add_frames(data.data)
            # Process chunk if enough frames accumulated
            while buffer.num_frames >= chunk_frames:
                chunk = buffer.pop_frames(chunk_frames)
                await self._process_chunk(chunk)

    async def _process_chunk(self, buffer: AudioBuffer):
        # Mark start of speech
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

        # Recognize with Whisper
        event = await self.stt._recognize_impl(buffer, language=self._opts.language)
        self._event_ch.send_nowait(event)

        # Mark end of speech
        self._speaking = False
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    async def aclose(self):
        await self._input_queue.put(self._FlushSentinel())
        await super().aclose()
