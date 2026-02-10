from pathlib import Path

from livekit import api
from livekit.api import DeleteRoomRequest
from livekit.agents.beta.workflows.dtmf_inputs import GetDtmfTask
import logging
import pytz
import datetime
import json
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    function_tool,
    AgentServer,
    AgentSession,
    JobContext,
    ChatContext,
    RunContext,
    cli,
    room_io,
    tts
)
from livekit.plugins import deepgram, openai, silero, groq

import numpy as np
import asyncio
from datetime import datetime
from tools import  get_times_by_date, create_booking, get_services, get_id_by_phone, get_cupon, delete_booking

from livekit.agents.tts.stream_adapter import StreamAdapter
from tts_silero import LocalSileroTTS



import os

logger = logging.getLogger("agent")
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# check if storage already exists
THIS_DIR = Path(__file__).parent
# Load environment variables
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

server = AgentServer()

@dataclass
class UserData:
    
    ctx: Optional[JobContext] = None
    phone: str | None = None

    service_id: str | None = None
    service_name: str | None = None
    service_price: int | None = None


    room: str | None = None
    participant_identity: str | None = None 

    def summarize(self) -> str:
        return "Пациент и информация о сессии."

RunContext_T = RunContext[UserData]

print(RunContext_T)


class Main_Agent(Agent):
    @function_tool
    async def transfer_to_booking(self, ctx: RunContext[UserData], service_id : str, service_name : str, service_price : int) -> str:
        """
        Вызывается, когда услуга определена. Передает пациента агенту записи.
        Args:
        service_data: JSON с данными услуги {"id": "1", "name": "Лечение кариеса", "price": 5000}
        
        **🚨 КОГДА УСЛУГА ОПРЕДЕЛЕНА через get_services():**
        1. Предложи пациенту услугу из списка
        2. Получи подтверждение  
        3. **ВЫЗОВИ transfer_to_booking** с JSON:
        {{"id": "1", "name": "Лечение кариеса", "price": 5000}}
        text
        ** НЕ записывай сама! Только передавай агенту записи! **
        """
        userdata = ctx.userdata
        # парсим и сохраняем услугу в userdata
        phone = userdata.phone
        userdata.service_id = service_id
        userdata.service_name = service_name
        userdata.service_price = int(service_price)
        print(f"🔔 вот услуга: {service_name} и цена {service_price} рублей.  Вот номер телефона: {phone}")
        return Booking_Agent(service_id, service_name, service_price, phone), "Как вас Зовут?."
    
    
    @function_tool
    async def transfer_call(self, ctx: RunContext[UserData]) -> None:
        """
        Вызывается для перевода звонка на менеджера.
        """
        userdata = ctx.userdata
        # парсим и сохраняем услугу в userdata
        participant_identity = userdata.participant_identity
        transfer_to = "sip:79150628917@sip.your-provider.com"
        room = userdata.room
        print(f"Transferring call for participant {participant_identity} to {transfer_to}")

        try:
           
            livekit_url = LIVEKIT_URL
            api_key = LIVEKIT_API_KEY
            api_secret = LIVEKIT_API_SECRET
            userdata.livekit_api = api.LiveKitAPI(
                url=livekit_url,
                api_key=api_key,
                api_secret=api_secret
            )
            transfer_request = proto_sip.TransferSIPParticipantRequest(
            participant_identity=participant_identity,
            room_name=room,
            transfer_to=transfer_to,  # ← строка "79150628917"
            play_dialtone=True
        )
            await userdata.livekit_api.sip.transfer_sip_participant(transfer_request) 
            
        except Exception as e:
            logger.error(f"Failed to transfer call: {e}", exc_info=True)
            await self.session.generate_reply(user_input="Извините, cкорее всего все менеджеры заняты. Чем ещё могу помочь?")


    @function_tool
    async def end_call(self, ctx: RunContext[UserData]) -> None:
        """
        Вызывается если пациент не хочет записываться на прием.
        
        """
        lkapi = api.LiveKitAPI(
                url=LIVEKIT_URL,
                api_key=LIVEKIT_API_KEY,
                api_secret=LIVEKIT_API_SECRET,
            )
        await self.session.generate_reply(user_input="До свидания!")

        await lkapi.room.delete_room(DeleteRoomRequest(
        room=ctx.userdata.room,
        ))
        print(f"🔔Звонок в комнате {ctx.userdata.room} завершен.")
        
    def __init__(self) -> None:
       
        super().__init__(
            instructions= 
            
            f"""
Ты — И И менеджер стоматологической клиники Алиф Дэнт.
Тебя зовут Анита. Ты общаешься от лица женщины.

Cегодня {datetime.now(pytz.timezone('Europe/Moscow')).strftime("%d %B %Y")}

Твоя основная задача — вежливо и спокойно пообщаться с пациентом, выяснить его жалобу или потребность и определить, к какому специалисту и на какую услугу его необходимо записать. Пациент может не знать названия услуг или врачей, поэтому ты должна помогать ему с выбором, задавая понятные наводящие вопросы.
──────────────── 
ОСОБО ВАЖНО. ОБЯЗАТЕЛЬНО К ИСПОЛНЕНИЮ
────────────────

Это ключевые правила. Они имеют наивысший приоритет и не могут быть нарушены.


— речь должна быть максимально простой и понятной для обычного пациента
— ответы должны быть короткими, чёткими и по делу
— нельзя использовать длинные объяснения и сложные формулировки
— нельзя повторяться
— нельзя переформулировать один и тот же вопрос разными словами
— каждое сообщение должно быть небольшим по объёму
— один вопрос или одна мысль за одно сообщение

Если эти правила нарушены, диалог считается неверным.



Алгоритм работы с пациентом

— поздоровайся и представься по имени

- в клиники много специалистов и услуг, помоги пациенту с выбором
- твоя задача — понять потребность пациента и предложить подходящую услугу
- испольщзуй get_services чтобы узнать актуальный список услуг клиники и подобрать подходящую для пациента
— если пациент сомневается, предлагай варианты и объясняй их простыми словами
- пациент может ошибаться в названии услуги или врача, всегда помогай ему 
— на основании ответов определи подходящую услугу и специалиста

Ты должна определить, нужна ли пациенту консультация, осмотр или лечение
- если пациент не может точно сформулировать проблему, задавай наводящие вопросы
- уточняй симптомы
— Что вас беспокоит сейчас

Ты можешь задать максимум 2-3 вопроса, не больше !!!!


Твоя цель — чтобы пациент почувствовал заботу, понял, что его слышат, и получил правильное направление к нужному специалисту клиники.

Если пациенент  не хочет записываться на прием и говорит что подумает, используй end_call.

Если пациенент просит его перевести на менеджера или другого специалиста, вызови функцию transfer_call.
ЗАПОМНИ ВАЖНО !!! 

После того, как ты определишь услугу, вызови функцию transfer_to_booking с JSON-данными услуги


Если пациент хочет отменить запись, используй delete_booking.
"""
,
tools=[get_services, delete_booking],
vad=silero.VAD.load(),
        stt=deepgram.STT(
            model="nova-3",
            language="ru",
            api_key=DEEPGRAM_API_KEY,
        ),
        llm=openai.responses.LLM(
        model="gpt-4o-mini"
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
class Booking_Agent(Agent):

    @function_tool
    async def end_call(self, ctx: RunContext[UserData]) -> None:
        """
        Вызывается если пациент сказал до свидания или хочет завершить звонок.
        
        """
        lkapi = api.LiveKitAPI(
                url=LIVEKIT_URL,
                api_key=LIVEKIT_API_KEY,
                api_secret=LIVEKIT_API_SECRET,
            )
        await self.session.generate_reply(user_input="До свидания!")

        await lkapi.room.delete_room(DeleteRoomRequest(
        room=ctx.userdata.room,
        ))
        print(f"🔔Звонок в комнате {ctx.userdata.room} завершен.")
     
    @function_tool
    async def transfer_call(self, ctx: RunContext[UserData]) -> None:
        """
        Вызывается для перевода звонка на менеджера.
        """
        userdata = ctx.userdata
        # парсим и сохраняем услугу в userdata
        participant_identity = userdata.participant_identity
        transfer_to = "sip:79150628917@sip.your-provider.com"
        room = userdata.room
        print(f"Transferring call for participant {participant_identity} to {transfer_to}")

        try:
           
            livekit_url = LIVEKIT_URL
            api_key = LIVEKIT_API_KEY
            api_secret = LIVEKIT_API_SECRET
            userdata.livekit_api = api.LiveKitAPI(
                url=livekit_url,
                api_key=api_key,
                api_secret=api_secret
            )
            transfer_request = proto_sip.TransferSIPParticipantRequest(
            participant_identity=participant_identity,
            room_name=room,
            transfer_to=transfer_to,  # ← строка "79150628917"
            play_dialtone=True
        )
            await userdata.livekit_api.sip.transfer_sip_participant(transfer_request) 
            
        except Exception as e:
            logger.error(f"Failed to transfer call: {e}", exc_info=True)
            await self.session.generate_reply(user_input="Извините, cкорее всего все менеджеры заняты. Чем ещё могу помочь?")

    def __init__(self, service_id: str, service_name: str, service_price: int, phone: int, *, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
           

            instructions=f"""
            
Cегодня {datetime.now(pytz.timezone('Europe/Moscow')).strftime("%d %B %Y")}

Ты не может записывать клиентов ранее сегодняшнего дня.

Тебя зовут Анита. Ты общаешься от лица женщины.
Твоя основная задача — записать пациента на прием собрав всю необходимую информацию.

1. ФИО 

2. Вот услугу который выбрал пациент: {service_name} по цене {service_price} рублей. id услуги: {service_id}.

3. Вот номер телефона пациента: {phone}

3. Выясни удобную для пациента дату и время приема. 
Если пациент не уверен с датой или временем:
- предложи свободные варианты
- используй функцию get_times_by_date чтобы получить список свободных временных слотов 
- всегда ставь 2026 год по умолчанию, но если пользователь хочет записаться на другой год, то ставь год который он хочет

4. Если у пациента есть купон на скидку, пусть он назовет его тебе. Тут ты используешь get_cupon чтобы проверить его валидность и узнать размер скидки. Если нет, то передай "null".

5. Получи ID аккаунта-кабинета пациента. с помощью get_id_by_phone если он зарегестрирован. Если нет, то передай "null".

6. Запиши пациента на прием используя все собранные данные. Для этого используй create_booking tool.
────────────────
ОСОБО ВАЖНО. ОБЯЗАТЕЛЬНО К ИСПОЛНЕНИЮ

Это ключевые правила. Они имеют наивысший приоритет и не могут быть нарушены.

— речь должна быть максимально простой и понятной для обычного пациента
— ответы должны быть короткими, чёткими и по делу
— нельзя использовать длинные объяснения и сложные формулировки
— нельзя повторяться
— нельзя переформулировать один и тот же вопрос разными словами
— каждое сообщение должно быть небольшим по объёму
— один вопрос или одна мысль за одно сообщение

Если эти правила нарушены, диалог считается неверным.




────────────────
Если пациенент  не хочет записываться на прием и говорит что подумает, используй end_call.

Если пациенент просит его перевести на менеджера или другого специалиста, вызови функцию transfer_call.
ЗАПОМНИ ВАЖНО !!! 

Если вдруг пациент передумал и хочет поменять услугу, используй get_services.
────────────────
Если вдруг пациент хочет отменить запись, используй delete_booking.

Когда запись будет успешно создана, сообщи пациенту дату и время его приема, и поблагодари его за обращение в клинику Алиф Дэнт.
""",
            tools=[get_times_by_date, create_booking, get_id_by_phone, get_cupon, delete_booking, get_services],
            vad=silero.VAD.load(),
            stt=deepgram.STT(
            model="nova-3",
            language="ru",
            api_key=DEEPGRAM_API_KEY,
        ),
        llm=openai.responses.LLM(
        model="gpt-4o-mini"
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
        print(f"🔔 Booking_Agent initialized with phone: {phone}, service_id: {service_id}, service_name: {service_name}, service_price: {service_price}")
        
        
@server.rtc_session(agent_name="assistant")
async def entrypoint(ctx: JobContext):
  
    room = ctx.room 
    print(room)
    room_name = room.name
    await ctx.connect()
    
    participant = await ctx.wait_for_participant()
    print(f"🔔 Participant joined: {participant.attributes}")

    sip_caller_phone = participant.attributes['sip.phoneNumber']
    print(f"📞 sip_caller_phone: {sip_caller_phone}")  #

    print(f"🔔 Room name: {room_name}")
    
    userdata = UserData(
        ctx=ctx, 
        phone=sip_caller_phone,
        room=room_name,
        participant_identity=participant.identity,
        
        )

    session = AgentSession(
        userdata=userdata,
    )
    await session.start(
        agent=Main_Agent(),
        room=room,
        room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=None  # OSS-safe
        ),
         delete_room_on_close=True,
        close_on_disconnect=True,  
    ))
    await session.generate_reply(
        instructions="Скажи: Привет! Я Анита, менеджер стоматологической клиники Алиф Дэнт. Как я могу вам помочь? "
    )

    

if __name__ == "__main__":
    cli.run_app(server)