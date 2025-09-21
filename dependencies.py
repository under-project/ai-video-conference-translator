import os
import sys
import subprocess
import platform

def install_dependencies():
    """Install dependencies berdasarkan platform"""
    system = platform.system().lower()

    print("Menginstall RNNoise...")
    
    # Dependencies dasar
    dependencies = [
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "openai-whisper>=20230314",
        "ollama>=0.1.0",
        "TTS>=0.21.0",
        "webrtcvad>=2.0.10",
        "librosa>=0.10.0",
        "pyannote.audio>=3.1.0"
    ]
    
    # Perintah instalasi
    if system == "windows":
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        
    elif system == "linux":
        # Untuk Linux - install system dependencies terlebih dahulu
        linux_deps = [
            "libsndfile1", "ffmpeg", "portaudio19-dev",
            "python3-dev", "python3-pip", "libopenblas-dev",
            "build-essential", "autoconf", "libtool"
        ]
        subprocess.check_call(["sudo", "apt-get", "update"])
        subprocess.check_call(["sudo", "apt-get", "install", "-y"] + linux_deps)
        
        # Install Python dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        
        # Install RNNoise
        try:
            subprocess.check_call(["git", "clone", "https://github.com/xiph/rnnoise.git"])
            os.chdir("rnnoise")
            subprocess.check_call(["./autogen.sh"])
            subprocess.check_call(["./configure"])
            subprocess.check_call(["make"])
            subprocess.check_call(["sudo", "make", "install"])
            os.chdir("..")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rnnoise"])
        except Exception as e:
            print(f"Error installing RNNoise: {e}")
        
    elif system == "darwin":  # macOS
        # Untuk macOS - install Homebrew jika belum ada
        try:
            subprocess.check_call(["brew", "--version"])
        except:
            print("Menginstall Homebrew...")
            subprocess.check_call('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"', shell=True)
        
        # Install system dependencies
        subprocess.check_call(["brew", "install", "ffmpeg", "portaudio", "autoconf", "automake", "libtool"])
        
        # Install Python dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        
        # Install RNNoise
        try:
            subprocess.check_call(["git", "clone", "https://github.com/xiph/rnnoise.git"])
            os.chdir("rnnoise")
            subprocess.check_call(["./autogen.sh"])
            subprocess.check_call(["./configure"])
            subprocess.check_call(["make"])
            subprocess.check_call(["sudo", "make", "install"])
            os.chdir("..")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rnnoise"])
        except Exception as e:
            print(f"Error installing RNNoise: {e}")
    
    print("Dependencies berhasil diinstall!")

if __name__ == "__main__":
    install_dependencies()