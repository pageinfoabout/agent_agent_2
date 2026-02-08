from flask import Flask, request
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import time
import uuid
from collections import deque, defaultdict
import threading
from transformers import pipeline
import torch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stt-server-key-2026'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Load Whisper model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-large-v3-turbo", 
    device=device
)

# Session state
audio_buffers = {}
item_timings = {}

@socketio.on('connect')
def connect():
    print(f"✅ STT Client CONNECTED: {request.sid}")
    audio_buffers[request.sid] = deque(maxlen=24000*10)  # 10s @ 24kHz
    item_timings[request.sid] = {}

@socketio.on('disconnect')
def disconnect():
    print(f"❌ STT Client DISCONNECTED: {request.sid}")
    audio_buffers.pop(request.sid, None)
    item_timings.pop(request.sid, None)

@socketio.on('message')
def handle_message(data):
    msg_type = data.get('type')
    print(f"📨 Received: {msg_type}")
    
    if msg_type == 'session.update':
        # REQUIRED handshake for stt.py
        emit('session.created', {'session_id': request.sid})
        emit('session.updated', {'session_id': request.sid})
        print(f"✅ Session configured: {request.sid}")
        
    elif msg_type == 'input_audio_buffer.append':
        # Store audio chunk
        audio_b64 = data['audio']
        audio_bytes = base64.b64decode(audio_b64)
        audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
        
        sid = request.sid
        audio_buffers[sid].extend(audio_np)
        
        item_id = data.get('item_id', str(uuid.uuid4()))
        
        # Send speech started
        emit('input_audio_buffer.speech_started', {
            'item_id': item_id,
            'audio_start_ms': 0
        })
        
        item_timings[sid][item_id] = {'start_ms': int(time.time() * 1000)}
        
        # Process if enough audio
        if len(audio_buffers[sid]) > 24000 * 1.5:  # 1.5 seconds
            threading.Thread(target=process_audio, 
                           args=(sid, item_id)).start()
    
    elif msg_type == 'input_audio_buffer.speech_stopped':
        item_id = data.get('item_id')
        sid = request.sid
        if sid in audio_buffers and audio_buffers[sid]:
            threading.Thread(target=process_audio, 
                           args=(sid, item_id)).start()

def process_audio(sid, item_id):
    """Process audio buffer with Whisper"""
    if sid not in audio_buffers or not audio_buffers[sid]:
        return
        
    audio_array = np.array(audio_buffers[sid])
    audio_buffers[sid].clear()
    
    try:
        # Transcribe
        result = pipe(audio_array, return_timestamps=False)
        transcript = result['text'].strip()
        
        # Calculate timing
        start_ms = item_timings[sid].get(item_id, {}).get('start_ms', 0)
        end_ms = int(time.time() * 1000)
        duration_ms = end_ms - start_ms
        
        # Send results
        socketio.emit('input_audio_buffer.speech_stopped', {
            'item_id': item_id,
            'audio_end_ms': duration_ms
        }, room=sid)
        
        if transcript:
            socketio.emit('conversation.item.input_audio_transcription.completed', {
                'item_id': item_id,
                'transcript': transcript,
                'words': [{'start': 0, 'end': duration_ms, 'word': transcript}]
            }, room=sid)
            print(f"✅ Transcribed: '{transcript}' (item: {item_id})")
        else:
            print(f"⚠️ Empty transcription (item: {item_id})")
            
    except Exception as e:
        print(f"❌ Transcription error: {e}")
    
    finally:
        if item_id in item_timings[sid]:
            del item_timings[sid][item_id]

if __name__ == '__main__':
    print("🚀 STT Server starting on http://0.0.0.0:5000")
    print("✅ Ready for LiveKit STT connections!")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
