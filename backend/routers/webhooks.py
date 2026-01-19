from livekit.api import webhook as lk_webhook  # ✅ ПРАВИЛЬНЫЙ ИМПОРТ
from livekit import api
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, APIRouter

load_dotenv()
app = FastAPI()

router = APIRouter(tags=["webhooks"])
# Глобально (после load_dotenv)
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

token_verifier = lk_webhook.TokenVerifier(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
webhook_receiver = lk_webhook.WebhookReceiver(token_verifier)

@router.post("/caller")
async def livekit_webhook(request: Request):
    body = await request.body()
    auth_header = request.headers.get("Authorization")

    event = webhook_receiver.receive_webhook(body, auth_header)
    return {"body": body, "event": event}