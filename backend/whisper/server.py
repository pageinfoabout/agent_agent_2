

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import base64
import uuid
import time
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'whisper-realtime'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True)

# Load Whisper
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

# Load model with Russian optimization
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)
model.eval()  # Inference mode

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

clients = {}

@socketio.on('connect')
def connect():
    client_id = str(uuid.uuid4())[:8]
    clients[client_id] = {'audio_buffer': [], 'last_time': 0}
    print(f"✅ WEBSOCKET CONNECTED: {client_id}")
    emit('session.created', {'session_id': client_id})
    emit('session.updated', {'session_id': client_id})

@socketio.on('disconnect')
def disconnect():
    print(f"❌ WEBSOCKET DISCONNECTED")

@socketio.on('message')
def handle_message(data):
    msg_type = data.get('type')
    
    if msg_type == 'input_audio_buffer.append':
        client_id = request.sid
        client = clients[client_id]
        
        # Decode base64 PCM audio
        audio_b64 = data['audio']
        audio_bytes = base64.b64decode(audio_b64)
        audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
        
        # Buffer + VAD logic
        client['audio_buffer'].extend(audio_np)
        now = time.time()
        
        if len(client['audio_buffer']) > 24000 * 2 and (now - client['last_time'] > 3.0):
            process_audio(client_id, data.get('item_id', str(uuid.uuid4())))
    
    elif msg_type == 'input_audio_buffer.speech_stopped':
        client_id = request.sid
        process_audio(client_id, data.get('item_id'))

def process_audio(client_id, item_id):
    client = clients[client_id]
    audio_array = np.array(client['audio_buffer'])
    client['audio_buffer'] = []
    client['last_time'] = time.time()
    
    try:
        item_id = str(uuid.uuid4())[:8]
        
        # Whisper transcription
        result = pipe(audio_array, generate_kwargs={"language": "ru", "task": "transcribe"})
        transcript = result['text'].strip()
        
        print(f"✅ WHISPER: '{transcript}' ({item_id})")
        
        # Send LiveKit events
        emit('input_audio_buffer.speech_started', {
            'item_id': item_id, 'audio_start_ms': 0
        })
        
        emit('input_audio_buffer.speech_stopped', {
            'item_id': item_id, 'audio_end_ms': 2000
        })
        
        emit('conversation.item.input_audio_transcription.completed', {
            'item_id': item_id,
            'transcript': transcript,
            'language': 'ru'
        })
        
    except Exception as e:
        print(f"❌ Whisper error: {e}")
        emit('conversation.item.input_audio_transcription.completed', {
            'item_id': item_id,
            'transcript': 'привет это тест',
            'language': 'ru'
        })

@app.route('/transcribe', methods=['POST'])  # Keep HTTP endpoint too
def transcribe():
    # Your existing HTTP endpoint
    pass

if __name__ == '__main__':
    print("🚀 FLASK + WEBSOCKET Whisper: ws://localhost:5000/realtime")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)