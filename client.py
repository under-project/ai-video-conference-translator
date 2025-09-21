import tkinter as tk
from tkinter import ttk, messagebox
import socket
import json
import threading
import sounddevice as sd
import numpy as np
from collections import deque

class TranslationClient:
    def __init__(self, root):
        self.root = root
        self.root.title("Realtime Speech-to-Speech Translation")
        self.root.geometry("600x500")
        
        # Variabel untuk koneksi dan pengaturan
        self.is_connected = False
        self.is_recording = False
        self.socket = None
        self.audio_buffer = deque(maxlen=48000)
        self.sample_rate = 16000
        
        # Setup GUI
        self.setup_gui()
        
        # Dapatkan daftar perangkat audio
        self.refresh_audio_devices()
    
    def setup_gui(self):
        """Setup antarmuka pengguna"""
        # Frame koneksi
        connection_frame = ttk.LabelFrame(self.root, text="Koneksi Server", padding=10)
        connection_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(connection_frame, text="Host:").grid(row=0, column=0, sticky="w")
        self.host_entry = ttk.Entry(connection_frame, width=20)
        self.host_entry.insert(0, "localhost")
        self.host_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(connection_frame, text="Port:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.port_entry = ttk.Entry(connection_frame, width=10)
        self.port_entry.insert(0, "12345")
        self.port_entry.grid(row=0, column=3, padx=5)
        
        self.connect_button = ttk.Button(connection_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=4, padx=(10, 0))
        
        # Frame pengaturan audio
        audio_frame = ttk.LabelFrame(self.root, text="Pengaturan Audio", padding=10)
        audio_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(audio_frame, text="Input Device:").grid(row=0, column=0, sticky="w")
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(audio_frame, textvariable=self.input_device_var, width=40)
        self.input_device_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(audio_frame, text="Output Device:").grid(row=1, column=0, sticky="w")
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(audio_frame, textvariable=self.output_device_var, width=40)
        self.output_device_combo.grid(row=1, column=1, padx=5, pady=5)
        
        self.refresh_devices_button = ttk.Button(audio_frame, text="Refresh Devices", command=self.refresh_audio_devices)
        self.refresh_devices_button.grid(row=2, column=1, pady=5, sticky="e")
        
        # Frame pengaturan bahasa
        language_frame = ttk.LabelFrame(self.root, text="Pengaturan Bahasa", padding=10)
        language_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(language_frame, text="Bahasa Sumber:").grid(row=0, column=0, sticky="w")
        self.source_lang_var = tk.StringVar()
        self.source_lang_combo = ttk.Combobox(language_frame, textvariable=self.source_lang_var, width=20)
        self.source_lang_combo['values'] = ('auto', 'id', 'en', 'ja', 'ko', 'ar')
        self.source_lang_combo.set('auto')
        self.source_lang_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(language_frame, text="Bahasa Tujuan:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.target_lang_var = tk.StringVar()
        self.target_lang_combo = ttk.Combobox(language_frame, textvariable=self.target_lang_var, width=20)
        self.target_lang_combo['values'] = ('en', 'id', 'ja', 'ko', 'ar')
        self.target_lang_combo.set('en')
        self.target_lang_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Frame kontrol
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        self.record_button = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording, state="disabled")
        self.record_button.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def refresh_audio_devices(self):
        """Refresh daftar perangkat audio yang tersedia"""
        try:
            devices = sd.query_devices()
            input_devices = []
            output_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device['name']))
                if device['max_output_channels'] > 0:
                    output_devices.append((i, device['name']))
            
            # Update comboboxes
            self.input_device_combo['values'] = [f"{idx}: {name}" for idx, name in input_devices]
            self.output_device_combo['values'] = [f"{idx}: {name}" for idx, name in output_devices]
            
            # Set default devices jika ada
            if input_devices:
                self.input_device_combo.set(f"{input_devices[0][0]}: {input_devices[0][1]}")
            if output_devices:
                self.output_device_combo.set(f"{output_devices[0][0]}: {output_devices[0][1]}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mendapatkan daftar perangkat audio: {e}")
    
    def get_device_id(self, device_str):
        """Extract device ID dari string pilihan"""
        if not device_str:
            return None
        try:
            return int(device_str.split(":")[0])
        except:
            return None
    
    def toggle_connection(self):
        """Toggle koneksi ke server"""
        if not self.is_connected:
            self.connect_to_server()
        else:
            self.disconnect_from_server()
    
    def connect_to_server(self):
        """Koneksi ke server"""
        try:
            host = self.host_entry.get()
            port = int(self.port_entry.get())
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((host, port))
            
            self.is_connected = True
            self.connect_button.config(text="Disconnect")
            self.record_button.config(state="normal")
            self.status_var.set(f"Connected to {host}:{port}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal terhubung ke server: {e}")
    
    def disconnect_from_server(self):
        """Putuskan koneksi dari server"""
        try:
            self.stop_recording()
            if self.socket:
                self.socket.close()
            
            self.is_connected = False
            self.connect_button.config(text="Connect")
            self.record_button.config(state="disabled")
            self.status_var.set("Disconnected")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memutuskan koneksi: {e}")
    
    def toggle_recording(self):
        """Toggle recording audio"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Mulai merekam dan mengirim audio ke server"""
        try:
            input_device = self.get_device_id(self.input_device_var.get())
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.status_var.set("Recording...")
            
            # Setup audio stream
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                device=input_device,
                blocksize=1024  # Buffer kecil untuk latency rendah
            )
            
            self.stream.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memulai recording: {e}")
            self.is_recording = False
            self.record_button.config(text="Start Recording")
    
    def stop_recording(self):
        """Hentikan recording"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.status_var.set("Recording stopped")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback untuk menangani data audio yang masuk"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Tambahkan audio ke buffer
        audio_data = indata[:, 0].copy()  # Ambil channel pertama
        self.audio_buffer.extend(audio_data)
        
        # Kirim data audio ke server jika terhubung dan buffer cukup penuh
        if self.is_connected and self.socket and len(self.audio_buffer) >= 4096:
            try:
                # Siapkan data untuk dikirim
                source_lang = self.source_lang_var.get()
                target_lang = self.target_lang_var.get()
                output_device = self.get_device_id(self.output_device_var.get())
                
                # Ambil data dari buffer
                audio_chunk = np.array(self.audio_buffer)
                self.audio_buffer.clear()  # Kosongkan buffer setelah mengambil data
                
                message = {
                    'audio_data': audio_chunk.tobytes().hex(),
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'output_device': output_device
                }
                
                # Kirim ke server
                self.socket.sendall(json.dumps(message).encode())
                
            except Exception as e:
                print(f"Error mengirim audio: {e}")
                self.disconnect_from_server()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationClient(root)
    root.mainloop()