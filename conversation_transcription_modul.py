import whisper
from TTS.utils.manage import ModelManager

def download_whisper_model():
    print("📥 Downloading Whisper model (base)...")
    whisper.load_model("base")
    print("✅ Whisper model downloaded.\n")

def download_coqui_tts_model():
    print("📥 Downloading Coqui TTS model (xtts_v2)...")
    manager = ModelManager()
    manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    print("✅ Coqui TTS model downloaded.\n")

if __name__ == "__main__":
    print("🚀 Starting model downloads...\n")
    download_whisper_model()
    download_coqui_tts_model()
    print("✅ All models downloaded successfully.")
