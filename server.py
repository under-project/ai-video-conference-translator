import io
import json
import socket
import threading
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from collections import deque
from context_aware_translator import ContextAwareTranslator
import ollama
import whisper
import webrtcvad  # Untuk VAD yang lebih baik
import os
import sys
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import librosa

# Coba import RNNoise, jika tidak tersedia gunakan fallback
try:
    from rnnoise import RNNoise
    RNNOISE_AVAILABLE = True
except ImportError:
    RNNOISE_AVAILABLE = False
    print("RNNoise tidak tersedia, menggunakan noise reduction alternatif")

class AdaptiveBuffer:
    def __init__(self, min_size=1024, max_size=8192, vad_aggressiveness=2):
        self.min_size = min_size
        self.max_size = max_size
        self.current_size = min_size
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.speech_speed = 1.0  # Normal speed
        self.speech_history = deque(maxlen=10)
        
    def update_buffer_size(self, audio_chunk, sample_rate=16000):
        """Update ukuran buffer berdasarkan VAD dan kecepatan bicara"""
        # Deteksi aktivitas suara
        is_speech = self.detect_speech(audio_chunk, sample_rate)
        
        # Hitung kecepatan bicara (kata per menit estimasi)
        if is_speech:
            speech_rate = self.estimate_speech_rate(audio_chunk)
            self.speech_history.append(speech_rate)
            
            # Update kecepatan bicara rata-rata
            if len(self.speech_history) > 0:
                self.speech_speed = np.mean(list(self.speech_history))
            
            # Sesuaikan ukuran buffer berdasarkan kecepatan bicara
            # Lebih cepat bicara = buffer lebih kecil
            speed_factor = max(0.5, min(2.0, 1.0 / self.speech_speed))
            new_size = int(self.current_size * speed_factor)
            self.current_size = max(self.min_size, min(self.max_size, new_size))
        else:
            # Tidak ada bicara, kembalikan ke ukuran normal
            self.current_size = self.min_size
            
        return self.current_size
    
    def detect_speech(self, audio_chunk, sample_rate):
        """Deteksi aktivitas suara menggunakan VAD"""
        try:
            # Konversi ke format yang diharapkan oleh webrtcvad (16-bit PCM)
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            return self.vad.is_speech(audio_int16.tobytes(), sample_rate)
        except:
            return False
    
    def estimate_speech_rate(self, audio_chunk):
        """Estimasi kecepatan bicara berdasarkan energi sinyal"""
        # Hitung energi sinyal
        energy = np.mean(audio_chunk**2)
        
        # Normalisasi energi untuk estimasi kecepatan
        # Nilai lebih tinggi = bicara lebih cepat
        speech_rate = min(2.0, max(0.5, energy * 10))
        return speech_rate

class RNNoiseProcessor:
    """Processor untuk noise suppression menggunakan RNNoise"""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.denoiser = None
        
        if RNNOISE_AVAILABLE:
            try:
                self.denoiser = RNNoise()
                print("RNNoise berhasil diinisialisasi")
            except Exception as e:
                print(f"Error inisialisasi RNNoise: {e}")
                self.denoiser = None
        else:
            print("RNNoise tidak tersedia, menggunakan metode alternatif")
    
    def process(self, audio_data):
        """Proses audio dengan RNNoise untuk noise suppression"""
        if self.denoiser is not None:
            try:
                # RNNoise expects 16-bit PCM audio
                audio_int16 = (audio_data * 32767).astype(np.int16)
                processed_audio = self.denoiser.process(audio_int16)
                return processed_audio.astype(np.float32) / 32767.0
            except Exception as e:
                print(f"Error processing dengan RNNoise: {e}")
                return audio_data
        else:
            # Fallback: menggunakan noise reduction sederhana
            return self.fallback_noise_reduction(audio_data)
    
    def fallback_noise_reduction(self, audio_data):
        """Metode fallback untuk noise reduction jika RNNoise tidak tersedia"""
        # Simple spectral gating noise reduction
        if len(audio_data) < 100:
            return audio_data
            
        # STFT untuk analisis frekuensi
        stft = librosa.stft(audio_data, n_fft=512, hop_length=128)
        magnitude, phase = librosa.magphase(stft)
        
        # Estimate noise from the first 100ms
        noise_frames = magnitude[:, :max(1, len(magnitude[0]) // 10)]
        noise_profile = np.mean(noise_frames, axis=1, keepdims=True)
        
        # Soft mask berdasarkan noise profile
        mask = magnitude / (magnitude + 2.0 * noise_profile)
        cleaned_stft = mask * stft
        
        # Inverse STFT
        cleaned_audio = librosa.istft(cleaned_stft, hop_length=128)
        
        # Pastikan panjang audio sama
        if len(cleaned_audio) > len(audio_data):
            cleaned_audio = cleaned_audio[:len(audio_data)]
        elif len(cleaned_audio) < len(audio_data):
            cleaned_audio = np.pad(cleaned_audio, (0, len(audio_data) - len(cleaned_audio)))
            
        return cleaned_audio.astype(np.float32)

class AudioProcessor:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_buffer = deque()
        self.is_recording = False
        self.adaptive_buffer = AdaptiveBuffer()
        self.whisper_model = None
        self.tts_model = None
        self.context_translator = None
        self.ollama_model = "llama2"  # Ganti dengan model yang sesuai
        self.target_lang = "en"
        self.rnnoise_processor = RNNoiseProcessor(sample_rate)
        
    def initialize_models(self):
        """Inisialisasi semua model AI"""
        print("Memuat model Whisper...")
        self.whisper_model = whisper.load_model("base")
        
        print("Memuat model Coqui TTS...")
        self.initialize_tts()
        
        print("Menyiapkan translator...")
        self.context_translator = ContextAwareTranslator()
        
        print("Semua model berhasil dimuat!")
    
    def initialize_tts(self):
        """Inisialisasi Coqui TTS"""
        try:
            # Setup model manager untuk Coqui TTS
            model_manager = ModelManager()
            
            # Download dan load model XTTS
            model_path, config_path, model_item = model_manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
            voc_path, voc_config_path, _ = model_manager.download_model("vocoder_models/en/ljspeech/hifigan_v2")
            
            # Inisialisasi synthesizer
            self.tts_synthesizer = Synthesizer(
                tts_checkpoint=model_path,
                tts_config_path=config_path,
                tts_speakers_file=None,
                tts_languages_file=None,
                vocoder_checkpoint=voc_path,
                vocoder_config=voc_config_path,
                encoder_checkpoint=None,
                encoder_config=None,
                use_cuda=torch.cuda.is_available(),
            )
            
            # Buat speaker reference
            self.setup_reference_speaker()
            
        except Exception as e:
            print(f"Error inisialisasi TTS: {e}")
            # Fallback ke TTS basic jika XTTS gagal
            from TTS.api import TTS
            self.tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
    
    def setup_reference_speaker(self):
        """Setup speaker reference untuk TTS"""
        try:
            # Buat sample audio referensi
            import wave
            import struct
            
            # Generate simple tone sebagai reference default
            sample_rate = 22050
            duration = 1.0  # seconds
            frequency = 440  # Hz

            # Generate sine wave            
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Simpan sebagai WAV file
            with wave.open("reference.wav", 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(struct.pack('<' + ('h' * len(audio_data)), *(audio_data * 32767).astype(np.int16)))
                
            self.reference_speaker = "reference.wav"
            
        except Exception as e:
            print(f"Error membuat reference speaker: {e}")
            self.reference_speaker = None
    
    def preprocess_audio(self, audio_data):
        """Pre-processing audio: RNNoise noise suppression lalu normalisasi volume"""
        try:
            # Konversi ke numpy array jika perlu
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.numpy()
            
            # 1. Noise suppression dengan RNNoise (atau fallback)
            audio_data = self.rnnoise_processor.process(audio_data)
            
            # 2. Normalisasi volume
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            return audio_data
        except Exception as e:
            print(f"Error pre-processing audio: {e}")
            return audio_data
    
    def transcribe_audio(self, audio_data):
        """Transkripsi audio ke teks menggunakan Whisper dengan BytesIO"""
        try:
            # Pre-processing audio (RNNoise + normalisasi)
            processed_audio = self.preprocess_audio(audio_data)
            
            # Konversi ke format yang diterima Whisper (mono, 16kHz)
            audio_np = np.array(processed_audio, dtype=np.float32)
            
            # Gunakan BytesIO untuk in-memory processing
            audio_buffer = io.BytesIO()
            np.save(audio_buffer, audio_np)
            audio_buffer.seek(0)
            
            # Transkripsi menggunakan Whisper
            result = self.whisper_model.transcribe(audio_np)  # Biarkan Whisper deteksi bahasa otomatis
            return result['text']
        except Exception as e:
            print(f"Error dalam transkripsi: {e}")
            return ""
    
    def text_to_speech(self, text, output_device=None):
        """Konversi teks ke speech menggunakan Coqui TTS"""
        if not text.strip():
            return
            
        try:
            # Generate speech menggunakan Coqui TTS
            if hasattr(self, 'tts_synthesizer'):
                # Gunakan XTTS synthesizer
                wav = self.tts_synthesizer.tts(
                    text=text,
                    speaker_name="default",
                    language=self.target_lang.split("_")[0],
                    speaker_wav=self.reference_speaker
                )
                audio_data = np.array(wav, dtype=np.float32)
            else:
                # Fallback ke TTS basic
                audio_data = self.tts_model.tts(
                    text=text,
                    speaker_wav="reference.wav",
                    language=self.target_lang.split("_")[0]
                )
            
            # Putar audio di output device yang dipilih
            if output_device is not None:
                sd.play(audio_data, samplerate=22050, device=output_device)
                sd.wait()
                
            return audio_data
        except Exception as e:
            print(f"Error dalam TTS: {e}")
    
    def process_audio_stream(self, audio_data, source_lang, target_lang, output_device=None):
        """Proses lengkap audio stream dari input ke output"""
        try:
            # Update ukuran buffer adaptif
            buffer_size = self.adaptive_buffer.update_buffer_size(audio_data, self.sample_rate)
            
            # Deteksi aktivitas suara
            if not self.adaptive_buffer.detect_speech(audio_data, self.sample_rate):
                return
            
            # Pastikan kita punya cukup data audio
            if len(audio_data) < buffer_size // 2:
                return
            
            # Transkripsi audio ke teks
            transcribed_text = self.transcribe_audio(audio_data)
            if not transcribed_text.strip():
                return
            
            print(f"Teks terdeteksi: {transcribed_text}")
            
            # Terjemahkan teks dengan konteks
            self.target_lang = target_lang
            translated_text = self.context_translator.translate_with_context(
                transcribed_text, self.ollama_model, target_lang
            )
            
            print(f"Teks diterjemahkan: {transcribed_text} -> {translated_text}")
            
            # Konversi teks ke speech
            self.text_to_speech(translated_text, output_device)
            
        except Exception as e:
            print(f"Error dalam proses audio: {e}")

class TranslationServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.audio_processor = AudioProcessor()
        self.is_running = False
        self.client_socket = None
        
    def start_server(self):
        """Jalankan server TCP"""
        self.audio_processor.initialize_models()
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        
        print(f"Server berjalan di {self.host}:{self.port}")
        self.is_running = True
        
        while self.is_running:
            try:
                self.client_socket, addr = server_socket.accept()
                print(f"Terhubung dengan client: {addr}")
                
                # Handle client dalam thread terpisah
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(self.client_socket,)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                print(f"Error server: {e}")
                break
        
        server_socket.close()
    
    def handle_client(self, client_socket):
        """Handle koneksi client"""
        try:
            while self.is_running:
                # Terima data audio dari client
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Decode data (format: JSON dengan metadata dan audio)
                try:
                    message = json.loads(data.decode())
                    audio_data = np.frombuffer(bytes.fromhex(message['audio_data']), dtype=np.float32)
                    source_lang = message['source_lang']
                    target_lang = message['target_lang']
                    output_device = message.get('output_device', None)
                    
                    # Proses audio
                    self.audio_processor.process_audio_stream(
                        audio_data, source_lang, target_lang, output_device
                    )
                    
                except Exception as e:
                    print(f"Error memproses data: {e}")
                    
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def stop_server(self):
        """Hentikan server"""
        self.is_running = False
        if self.client_socket:
            self.client_socket.close()
if __name__ == "__main__":
    server = TranslationServer()
    server.start_server()