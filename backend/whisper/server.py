from flask import Flask, request, Response
import asyncio
import base64
import json
import numpy as np
from transformers import pipeline
import torch
import threading
from collections import defaultdict
import time

app = Flask(__name__)

# Load Whisper (lightweight)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", 
                model="openai/whisper-large-v3-turbo", 
                device=device)

# Session state
sessions = defaultdict(dict)

@app.route('/realtime', websocket=True)  # ← THIS IS WHAT stt.py EXPECTS
def realtime():
    ws = request.environ['wsgi.websocket']
    
    # Wait for session.update from stt.py
    while True:
        message = ws.receive()
        if message is None:
            break
            
        data = json.loads(message)
        if data.get('type') == 'session.update':
            # Respond with required handshake
            ws.send(json.dumps({
                "type": "session.created",
                "session_id": "sess_123"
            }))
            ws.send(json.dumps({
                "type": "session.updated", 
                "session_id": "sess_123"
            }))
            print("✅ STT Session configured")
            break
    
    # Main audio processing loop
    buffer = []
    item_id = None
    
    while True:
        try:
            message = ws.receive()
            if message is None:
                break
                
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'input_audio_buffer.append':
                # Decode 24kHz PCM
                audio_b64 = data['audio']
                audio_bytes = base64.b64decode(audio_b64)
                audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
                buffer.extend(audio_np)
                
                if item_id is None:
                    item_id = data.get('item_id', 'item_1')
                    # Send speech started
                    ws.send(json.dumps({
                        "type": "input_audio_buffer.speech_started",
                        "item_id": item_id,
                        "audio_start_ms": 0
                    }))
                
                # Process if >1.5s audio
                if len(buffer) > 24000 * 1.5:
                    threading.Thread(target=process_audio, args=(ws, buffer, item_id)).start()
                    buffer = []
            
            elif msg_type == 'input_audio_buffer.speech_stopped':
                if buffer and item_id:
                    threading.Thread(target=process_audio, args=(ws, buffer, item_id)).start()
                    buffer = []
                
        except Exception as e:
            print(f"WS Error: {e}")
            break
    
    ws.close()

def process_audio(ws, audio_buffer, item_id):
    """Process audio with Whisper in background thread"""
    if not audio_buffer:
        return
        
    audio_array = np.array(audio_buffer)
    
    # Transcribe
    result = pipe(audio_array)
    transcript = result['text'].strip()
    
    # Send final events
    ws.send(json.dumps({
        "type": "input_audio_buffer.speech_stopped",
        "item_id": item_id,
        "audio_end_ms": 1500
    }))
    
    if transcript:
        ws.send(json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "transcript": transcript
        }))
        print(f"✅ Transcribed: '{transcript}'")
    else:
        print("⚠️ No transcription")

@app.route('/audio/transcriptions', methods=['POST'])  # Batch fallback
def batch_transcribe():
    if 'file' not in request.files:
        return {'error': 'No file'}, 400
    
    audio_file = request.files['file']
    audio_data = audio_file.read()
    
    result = pipe(audio_data)
    return {'text': result['text']}

if __name__ == '__main__':
    print("🚀 STT Server: ws://localhost:5000/realtime")
    app.run(host='0.0.0.0', port=5000, debug=True)
