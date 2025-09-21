class ContextAwareTranslator:
    def __init__(self, max_context_length=5):
        self.context_history = []
        self.max_context_length = max_context_length
        self.current_context = ""
    
    def update_context(self, new_text):
        """Update konteks dengan teks baru"""
        if not new_text.strip():
            return self.current_context
            
        # Tambahkan teks baru ke history
        self.context_history.append(new_text.strip())
        
        # Pertahankan ukuran history sesuai batas
        if len(self.context_history) > self.max_context_length:
            self.context_history.pop(0)
            
        # Perbarui konteks saat ini
        self.current_context = " ".join(self.context_history)
        return self.current_context
    
    def get_context(self):
        """Dapatkan konteks saat ini"""
        return self.current_context
    
    def clear_context(self):
        """Bersihkan konteks"""
        self.context_history = []
        self.current_context = ""
        return self.current_context
    
    def translate_with_context(self, text, ollama_model, target_lang):
        """Terjemahkan teks dengan mempertimbangkan konteks"""
        if not text.strip():
            return ""
            
        # Siapkan prompt dengan konteks
        if self.current_context:
            prompt = f"""
            Konteks sebelumnya: {self.current_context}
            Terjemahkan kalimat berikut ke {target_lang}: "{text}"
            Pastikan terjemahan sesuai dengan konteks dan terdengar alami.
            Hanya kembalikan terjemahannya saja tanpa tambahan apapun.
            """
        else:
            prompt = f"""
            Terjemahkan kalimat berikut ke {target_lang}: "{text}"
            Hanya kembalikan terjemahannya saja tanpa tambahan apapun.
            """
        
        # Gunakan Ollama untuk terjemahan
        try:
            response = ollama.chat(model=ollama_model, messages=[
                {
                    'role': 'system',
                    'content': 'Anda adalah penerjemah yang ahli. Terjemahkan teks dengan mempertimbangkan konteks yang diberikan.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            translated_text = response['message']['content'].strip()
            
            # Perbarui konteks dengan teks asli (bukan terjemahan)
            self.update_context(text)
            
            return translated_text
        except Exception as e:
            print(f"Error dalam terjemahan: {e}")
            return text  # Fallback ke teks asli jika terjemahan gagal