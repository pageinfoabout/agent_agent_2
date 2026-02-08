from pathlib import Path
import os
import logging
import pytz
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import numpy as np

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    function_tool,
    AgentServer,
    JobContext,
    ChatContext,
    RunContext,
    cli,
    room_io
)
from livekit.plugins import silero, openai

from tts_silero import LocalSileroTTS
from tools import get_times_by_date, create_booking, get_services, get_id_by_phone, get_cupon, delete_booking

logger = logging.getLogger("agent")
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

server = AgentServer()

# --- Load Whisper large locally ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# --- STT Adapter ---
class LocalWhisperSTT:
    def __init__(self, pipeline):
        self.pipe = pipeline

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        result = self.pipe(audio, sampling_rate=sample_rate)
        return result["text"]

stt_adapter = LocalWhisperSTT(whisper_pipe)

# --- User Data ---
@dataclass
class UserData:
    personas: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    ctx: Optional[JobContext] = None

    phone: Optional[str] = None
    service_id: Optional[str] = None
    service_name: Optional[str] = None
    service_price: Optional[int] = None

RunContext_T = RunContext[UserData]

# --- Main Agent ---
class Main_Agent(Agent):
    @function_tool
    async def transfer_to_booking(
        self, ctx: RunContext[UserData], service_id: str, service_name: str, service_price: int
    ):
        userdata = ctx.userdata
        phone = userdata.phone
        userdata.service_id = service_id
        userdata.service_name = service_name
        userdata.service_price = int(service_price)
        print(f"🔔 Service: {service_name}, Price: {service_price}, Phone: {phone}")
        return Booking_Agent(service_id, service_name, service_price, phone), "Как вас зовут?"

    def __init__(self):
        super().__init__(
            instructions=f"""
Ты — ИИ менеджер стоматологической клиники Алиф Дэнт. Тебя зовут Анита, общаешься от лица женщины.
Сегодня {datetime.now(pytz.timezone('Europe/Moscow')).strftime("%d %B %Y")}

Задача: выяснить жалобу пациента, определить услугу и вызвать transfer_to_booking.
""",
            tools=[get_services],
            vad=silero.VAD.load(),
            stt=stt_adapter,
            llm=openai.LLM.with_deepseek(
                model="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
                api_key=DEEPSEEK_API_KEY,
                temperature=0.2,
                top_p=0.3,
            ),
            tts=LocalSileroTTS(
                language="ru",
                model_id="v5_ru",
                speaker="baya",
                device="cpu",
                sample_rate=48000,
                put_accent=True,
                put_yo=True,
                put_stress_homo=False,
                put_yo_homo=True,
            ),
        )

# --- Booking Agent ---
class Booking_Agent(Agent):
    def __init__(self, service_id: str, service_name: str, service_price: int, phone: str, *, chat_ctx: Optional[ChatContext] = None):
        super().__init__(
            instructions=f"""
Ты — Анита, специалист по записи пациентов. Сегодня {datetime.now(pytz.timezone('Europe/Moscow')).strftime("%d %B %Y")}.
Пациент выбрал услугу: {service_name}, цена: {service_price}, id: {service_id}, телефон: {phone}.
Выясни дату и время приёма, проверь купон, получи ID кабинета и запиши через create_booking.
""",
            tools=[get_times_by_date, create_booking, get_id_by_phone, get_cupon, delete_booking],
            vad=silero.VAD.load(),
            stt=stt_adapter,
            llm=openai.LLM.with_deepseek(
                model="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
                api_key=DEEPSEEK_API_KEY,
                temperature=0.3,
                top_p=0.5,
            ),
            tts=LocalSileroTTS(
                language="ru",
                model_id="v5_ru",
                speaker="baya",
                device="cpu",
                sample_rate=48000,
                put_accent=True,
                put_yo=True,
                put_stress_homo=False,
                put_yo_homo=True,
            ),
            chat_ctx=chat_ctx,
        )

# --- LiveKit RTC Session ---
@server.rtc_session(agent_name="assistant")
async def entrypoint(ctx: JobContext):
    room = ctx.room
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    sip_caller_phone = participant.attributes.get('sip.phoneNumber')
    print(f"🔔 Participant joined: {sip_caller_phone}")

    userdata = UserData(ctx=ctx, phone=sip_caller_phone)
    agent_instance = Main_Agent(stt=stt_adapter)

    # Async TTS queue to avoid overlapping audio
    tts_queue = asyncio.Queue()

    async def tts_worker():
        while True:
            text = await tts_queue.get()
            try:
                audio_out = await agent_instance.tts.synthesize(text)
                await room.send_audio(audio_out)
            except Exception as e:
                print("TTS error:", e)
            tts_queue.task_done()

    asyncio.create_task(tts_worker())

    greeting = "Клиника «Алиф Дэнт». Здравствуйте, как я могу вам помочь?"
    await tts_queue.put(greeting)

    # --- Real-time audio loop ---
    async for audio_chunk in room.audio_stream():  # yields float32 np arrays
        speech_segments = agent_instance.vad.split(audio_chunk)
        for segment in speech_segments:
            try:
                text = await agent_instance.stt.transcribe(segment)
                if text.strip():
                    print("User said:", text)
                    response = await agent_instance.run(text)
                    print("Agent responds:", response)
                    await tts_queue.put(response)
            except Exception as e:
                print("Agent loop error:", e)

if __name__ == "__main__":
    cli.run_app(server)
