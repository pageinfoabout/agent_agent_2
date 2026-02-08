import asyncio
import websockets
import json
import uuid
import time
import random

clients = {}
phrases = [
    "привет я готов помочь как ваши дела",
    "да конечно что вас интересует", 
    "понимаю расскажите подробнее",
    "это можно быстро решить",
    "интересный вопрос давайте разберемся",
    "абсолютно согласен",
    "отлично давайте продолжим"
]

async def stt_handler(websocket):
    client_id = str(uuid.uuid4())[:8]
    clients[client_id] = {'ws': websocket, 'last_transcript': '', 'last_time': 0}
    print(f"✅ SMART STT {client_id} CONNECTED")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'session.update':
                await websocket.send(json.dumps({"type": "session.created", "session_id": client_id}))
                await websocket.send(json.dumps({"type": "session.updated", "session_id": client_id}))
                print(f"✅ Session: {client_id}")
                
            elif msg_type == 'input_audio_buffer.append':
                item_id = data.get('item_id', str(uuid.uuid4())[:8])
                client = clients[client_id]
                
                # Only transcribe every 3+ seconds (realistic VAD)
                now = time.time()
                if now - client['last_time'] > 3.0:
                    # New realistic phrase
                    transcript = random.choice(phrases)
                    client['last_transcript'] = transcript
                    client['last_time'] = now
                    
                    print(f"📤 SENDING: '{transcript}' ({item_id})")
                    
                    # Speech events
                    await websocket.send(json.dumps({
                        "type": "input_audio_buffer.speech_started",
                        "item_id": item_id, "audio_start_ms": 0
                    }))
                    await asyncio.sleep(0.5)
                    
                    await websocket.send(json.dumps({
                        "type": "input_audio_buffer.speech_stopped", 
                        "item_id": item_id, "audio_end_ms": 500
                    }))
                    
                    # Transcription
                    await websocket.send(json.dumps({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item_id,
                        "transcript": transcript,
                        "language": "ru"
                    }))
                else:
                    print(f"⏭️ Skipping duplicate ({now-client['last_time']:.1f}s)")
                
    except Exception as e:
        print(f"❌ {client_id}: {e}")
    finally:
        clients.pop(client_id, None)

async def main():
    print("🚀 SMART STT: Realistic Russian conversation!")
    server = await websockets.serve(stt_handler, "0.0.0.0", 5000)
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())