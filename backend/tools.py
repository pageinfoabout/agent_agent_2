

import os

from dotenv import load_dotenv
from livekit.agents import llm

from config.db import supabase
import logging
import json


logger = logging.getLogger("tools")
load_dotenv()

# Глобальный lkapi (один на все tools)
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")



@llm.function_tool
async def get_times_by_date(date: str) -> str:
    """
    Возвращает список уже занятых записей к врачу для указанной даты.

    В течение одного дня доступно 12 временных слотов:
    с 09:00 до 20:00.


    Функция используется для проверки занятости времени на выбранную дату.
    Она показывает, на какое время записи к врачу в поликлинике уже забронирован и недоступны
    для записи, а все остальные слоты считаются свободными и могут быть
    забронированы пациентом.

    :param date: По умлочанию всегда ставь 2026 год, но если пользователь хочет записаться на другой год, то ставь год который он хочет
    :return: Строка или структура данных со списком забронированных записей
    """

    response = supabase.table("bookings") \
        .select("date, time") \
        .eq("date", date) \
        .execute()
    print(f"Response. : {response}")

    if not response.data:
        return "На эту дату записей нет"
        
    return json.dumps(response.data, ensure_ascii=False)




@llm.function_tool
async def get_services() -> str:
    """
    Возвращает список услуг которые предоставляет поликлиника


    :return: Список услуг в формате JSON
    id = это название услуги по английскому языку (например: serv-orthodontic-maintenance)
    name = это название услуги по русскому языку
    price = это цена услуги

    :example:
    [
        {
            "id": "serv-orthodontic-maintenance",
            "name": "Услуги по обслуживанию ортодонтических аппаратов",
            "price": 1100
        },
    ]
       """
    response = supabase.table("services") \
        .select("*") \
        .execute()
    print(f"Response: {response}")
    if response.data:
        return json.dumps(response.data, ensure_ascii=False)
    else:
        return json.dumps({"error": "Не удалось получить услуги или таблица пуста"}, ensure_ascii=False)



@llm.function_tool
async def get_id_by_phone(sip_caller_phone: str) -> str:
    """

    Возвращает ID пользователя по номеру телефона
    убери все символы кроме цифр
    и номер телефона должен быть в формате 79000000000
    :param phone: Номер телефона пользователя
    :return: ID пользователя ( если нет id возращаешь null)
    :example:
    79000000000
    """
    response = supabase.table("users") \
        .select("id") \
        .eq("number", sip_caller_phone) \
        .execute()
    print(f"Response: {response}")
    if response.data:
        return json.dumps(response.data, ensure_ascii=False)
    else:
        return json.dumps({"error": "у пользователя нет id "}, ensure_ascii=False)


@llm.function_tool
async def get_cupon(cupon_name: str) -> str:
    """
    Возвращает информацию о купоне по его названию
    :param cupon_name: Название купона
    :return: Информация о купоне
    :example:
    "10% скидка"
    """
    response = supabase.table("cupons") \
        .select("*") \
        .eq("cupon_name", cupon_name) \
        .execute()
    print(f"Response: {response}")
    if response.data:
        return json.dumps(response.data, ensure_ascii=False)
    else:
        return json.dumps({"error": "Такого купона не существует" }, ensure_ascii=False)






@llm.function_tool
async def delete_booking(sip_caller_phone: str, date: str, time: str) -> str:
    """
    Удаляет запись по его ID
    :param phone: sip_caller_phone
    :param date: Дата приёма
    :param time: Время приёма
    :return: Сообщение об успешном удалении
    :example:
    79000000000
    2026-01-15
    14:00
    """
    phone_with_plus = f"+{sip_caller_phone}" 
    response = supabase.table("bookings").delete().eq("phone", phone_with_plus).eq("date", date).eq("time", time).execute()
    print(f"Response: {response}")
    if response.data:
        return json.dumps(response.data, ensure_ascii=False)
    else:
        return json.dumps({"error": "Запись не найдена" }, ensure_ascii=False)





@llm.function_tool
async def create_booking(name: str, sip_caller_phone: str, date: str, time: str, service_id: str, service_name: str, service_price: int, cabinet_id: str, cupon_name: str, discount_percent: int) -> str:
    """

        
        Ты вызываешь функцию создания записи только после того как пользователь подтвердил данные и сказал "да"
        если пользователь сказал "нет", то НЕ вызывай функцию.

        Перед вызовом функции ты ОБЯЗАН убедиться, что получены все обязательные данные.

        ---

        # Обязательные поля (без них функцию вызывать нельзя)

        1. name — имя пациента
        2. phone — sip_caller_phone
        3. date — дата приёма
        4. time — время приёма
        5. services — минимум одна выбранная услуга
        7. cupon_name — название купона
        если купона нет, то cupon_name = null

        ---

        # Дополнительные поля

        - cupon_name — название купона

        Правила работы с купоном:
        - если пациент называет купон → передай его строкой
        - если пациент говорит, что купона нет или он не знает → передай `null`
        - никогда не выдумывай купон

        ---

        # Формат услуги (services)

        Каждая услуга передаётся в виде объекта:
        {
        "id": "service_id" (например: serv-orthodontic-maintenance),
        "name": "service_name" (например: Услуги по обслуживанию ортодонтических аппаратов),
        "price": "service_price" (например: 1100)
        }

        В массиве services должна быть минимум одна услуга.

        ---

        # Логика работы

        1. Собери данные у пациента пошагово
        2. Повтори данные перед созданием записи
        3. Только после подтверждения пациента вызывай функцию
        4. Передай все данные строго в соответствии со схемой

        ---

        # Пример подтверждения перед вызовом функции

        "Подтверждаю запись:
        Имя: Анна  
        Телефон: +7 900 000 00 00  
        Дата: 15 января  
        Время: 14:00  
        Услуга: Лечение кариеса  
        Купон: без купона  

        Всё верно?"

        ---

        # Пример вызова функции (логика)

        name = "Анна"
        phone = "+79000000000"
        date = "2026-01-15"
        time = "14:00"
        services = [
        {
            "id": "1",
            "name": "Лечение кариеса",
            "price": 5000
        }
        ]
        cupon_name = null

    """
    

    if cabinet_id == "null":
        cabinet_id = None
    if cupon_name == "null":
        cupon_name = None
    if discount_percent == "null":
        discount_percent = None
        
    phone_with_plus = f"+{sip_caller_phone}" 
    payload = {
        "name": name,
        "phone": phone_with_plus,
        "date": date,
        "time": time,
        "services": [{
                      "id": service_id, 
                      "name": service_name, 
                      "price": service_price
                      }
        ],
        "total": service_price,
        "status": "new",
        "cabinet_id":  cabinet_id,
        "discount_percent":  discount_percent,
        "cupon_name": cupon_name,
      
    }
    response = supabase.table("bookings").insert(payload).execute()

    print(f"Response: {response}")
    print(f"Response error: {response.data}")

    if response.data:
        return json.dumps(response.data, ensure_ascii=False)
    
    if response.error:
        return json.dumps({"error": response.error.message + "Не удалось создать запись, попробуйте еще раз" }, ensure_ascii=False)
    else:
        return json.dumps({"error": "Не удалось создать запись, попробуйте еще раз" }, ensure_ascii=False)
    
    






