# FitTurkAI Backend

Bu klasör, FitTurkAI uygulamasının backend API'sini içerir. FastAPI ve Hugging Face Space API'si kullanarak AI modeliyle iletişim kurar.

## 📋 İçindekiler

- [⚙️ Kurulum](#️-kurulum)
- [🚀 Çalıştırma](#-çalıştırma)
- [🔧 Konfigürasyon](#-konfigürasyon)
- [📡 API Endpoints](#-api-endpoints)
- [🔒 Güvenlik](#-güvenlik)
- [🐛 Hata Giderme](#-hata-giderme)
- [📊 Performans](#-performans)

## ⚙️ Kurulum

### 📋 Gereksinimler

- **Python**: 3.9 veya üzeri
- **PIP**: Python paket yöneticisi
- **İnternet**: Hugging Face Space API erişimi için

### 🔧 Detaylı Kurulum

1. **📁 Backend klasörüne gidin:**
   ```bash
   cd "FitTürkAI Web Demo/backend"
   ```

2. **🐍 Python sanal ortamı oluşturun (önerilen):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **📦 Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **🔍 Kurulum doğrulaması:**
   ```bash
   python -c "
   import fastapi
   import gradio_client
   import uvicorn
   print('✅ Tüm bağımlılıklar başarıyla yüklendi!')
   "
   ```

## 🚀 Çalıştırma

### 🖥️ Geliştirme Modu

```bash
# Basit çalıştırma
python main.py

# Detaylı çalıştırma
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Debug modu
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

### 🌐 Production Modu

```bash
# Production için optimize edilmiş
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# SSL ile güvenli çalıştırma
uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem
```

Backend API'si varsayılan olarak `http://localhost:8000` adresinde çalışacaktır.

## 🔧 Konfigürasyon

### 🌍 Environment Variables

`.env` dosyası oluşturun ve aşağıdaki değişkenleri ekleyin:

```env
# API Konfigürasyonu
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Hugging Face Konfigürasyonu
HUGGING_FACE_SPACE=AIYildiz/AIYildizFitTurkAI
HUGGING_FACE_TOKEN=your_optional_token_here

# CORS Ayarları
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
CORS_ALLOW_CREDENTIALS=True

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Rate Limiting (gelecek özellik)
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Cache Ayarları (gelecek özellik)
CACHE_ENABLED=True
CACHE_TTL=300
```

### ⚙️ Model Konfigürasyonu

`main.py` dosyasında model ayarlarını değiştirebilirsiniz:

```python
# Farklı Hugging Face Space kullanımı
HUGGING_FACE_SPACE = "your-username/your-space-name"

# Timeout ayarları
GRADIO_TIMEOUT = 60  # saniye

# API yanıt ayarları
MAX_RESPONSE_LENGTH = 2000  # karakter
DEFAULT_CONTEXT_LENGTH = 4096
```

## 📡 API Endpoints

### 🔄 POST /chat
Kullanıcı mesajını alır ve Hugging Face Space'teki AI asistanından yanıt döner.

**Request Headers:**
```http
Content-Type: application/json
Accept: application/json
```

**Request Body:**
```json
{
  "soru": "Kullanıcının mesajı (zorunlu)",
  "gecmis": "Sohbet geçmişi (opsiyonel, string formatında)",
  "max_length": 1000,  // Opsiyonel, maksimum yanıt uzunluğu
  "temperature": 0.7   // Opsiyonel, yaratıcılık seviyesi
}
```

**Success Response (200):**
```json
{
  "cevap": "AI asistanının yanıtı",
  "status": "success",
  "response_time": 2.34,  // saniye
  "token_count": 156,      // yaklaşık token sayısı
  "model_info": {
    "space": "AIYildiz/AIYildizFitTurkAI",
    "version": "1.0.0"
  }
}
```

**Error Response (400/500):**
```json
{
  "error": "Hata açıklaması",
  "status": "error",
  "error_code": "INVALID_INPUT",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### ✅ GET /health
Sistem durumunu kontrol eder.

**Response (200):**
```json
{
  "status": "healthy",
  "api": "FitTürkAI Backend",
  "version": "1.0.0",
  "model": "Hugging Face Space",
  "uptime": 3600,  // saniye
  "last_model_check": "2024-01-15T10:30:00Z",
  "memory_usage": "245MB",
  "dependencies": {
    "fastapi": "0.104.1",
    "gradio_client": "0.8.1",
    "python": "3.11.5"
  }
}
```

### 📊 GET /metrics (gelecek özellik)
API kullanım istatistiklerini döner.

### 🔄 GET /models (gelecek özellik)  
Mevcut AI modellerinin listesini döner.

## 🔒 Güvenlik

### 🛡️ CORS Ayarları

```python
# Güvenlik için CORS ayarlarını production'da sınırlandırın
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Sadece frontend
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 🔐 Rate Limiting (gelecek özellik)

```python
# API kullanımını sınırlandırma
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")  # Dakikada 10 istek
async def chat_endpoint(request: Request, ...):
    pass
```

### 🔍 Input Validation

```python
# Girdi doğrulama ve temizleme
def sanitize_input(text: str) -> str:
    """Kullanıcı girdisini temizle ve doğrula"""
    if len(text) > 2000:
        raise ValueError("Mesaj çok uzun (max 2000 karakter)")
    
    # Tehlikeli karakterleri temizle
    cleaned = re.sub(r'[<>"\']', '', text)
    return cleaned.strip()
```

## 🐛 Hata Giderme

### ❗ Yaygın Hatalar ve Çözümleri

#### 1. **Port zaten kullanımda hatası**
```bash
# Hata: Port 8000 already in use
# Çözüm: Farklı port kullanın
uvicorn main:app --port 8001

# Veya port'u kapatan process'i bulun
lsof -ti:8000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8000   # Windows
```

#### 2. **Hugging Face Space bağlantı hatası**
```bash
# Hata: Connection timeout
# Çözüm 1: İnternet bağlantısını kontrol edin
ping huggingface.co

# Çözüm 2: Farklı Space deneyin
HUGGING_FACE_SPACE = "microsoft/DialoGPT-medium"

# Çözüm 3: Timeout süresini artırın
GRADIO_TIMEOUT = 120  # 2 dakika
```

#### 3. **ModuleNotFoundError**
```bash
# Hata: ModuleNotFoundError: No module named 'fastapi'
# Çözüm: Bağımlılıkları yeniden yükleyin
pip install -r requirements.txt --force-reinstall

# Sanal ortam aktif mi kontrol edin
which python  # /path/to/venv/bin/python olmalı
```

#### 4. **CORS hatası**
```bash
# Hata: CORS policy error
# Çözüm: Frontend URL'ini CORS ayarlarına ekleyin
allow_origins=["http://localhost:3000", "https://yourdomain.com"]
```

#### 5. **JSON decode hatası**
```bash
# Hata: JSONDecodeError
# Çözüm: Request body formatını kontrol edin
Content-Type: application/json
{
  "soru": "mesajınız"  # String olmalı
}
```

### 🔍 Debug Modu

Debug modunu aktifleştirmek için:

```python
# main.py dosyasında
import logging
logging.basicConfig(level=logging.DEBUG)

# Veya environment variable ile
export LOG_LEVEL=DEBUG
python main.py
```

### 📝 Log Analizi

```bash
# Logları takip etme
tail -f api.log

# Hata loglarını filtreleme
grep "ERROR" api.log

# Son 100 satırı görme
tail -n 100 api.log
```

## 📊 Performans

### ⚡ Optimizasyon Önerileri

1. **Response Caching** (gelecek özellik):
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_ai_response(question: str) -> str:
    return get_ai_response(question)
```

2. **Async Database Operations**:
```python
# Gelecekte database entegrasyonu için
import asyncpg

async def save_chat_async(user_id: int, message: str):
    pass
```

3. **Background Tasks**:
```python
from fastapi import BackgroundTasks

@app.post("/chat")
async def chat(background_tasks: BackgroundTasks):
    background_tasks.add_task(log_conversation, user_id, message)
```

### 📈 Monitoring

```python
# Performans metrikleri toplama
import time
import psutil

def get_system_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    }
```

### 🎯 Benchmark Sonuçları

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **Ortalama Yanıt Süresi** | ~2-5 saniye | Hugging Face Space'e bağlı |
| **Max Eşzamanlı İstek** | 10 | Ücretsiz tier sınırı |
| **Bellek Kullanımı** | ~100-200MB | FastAPI + Gradio Client |
| **Startup Süresi** | ~3 saniye | Cold start |

## 🔮 Gelecek Özellikler

### v1.1
- [ ] Rate limiting implementasyonu
- [ ] Response caching sistemi
- [ ] Detaylı logging ve monitoring
- [ ] Database entegrasyonu

### v1.2
- [ ] Authentication & authorization
- [ ] Multi-model support
- [ ] WebSocket real-time chat
- [ ] File upload desteği

### v2.0
- [ ] Microservices mimarisi
- [ ] Kubernetes deployment
- [ ] Advanced analytics
- [ ] Auto-scaling

---

## 📞 Destek

Sorunlarınız için:
- 🐛 **Bug Report**: [GitHub Issues](https://github.com/aiyildiz/fitturkai/issues)
- 💡 **Feature Request**: [GitHub Discussions](https://github.com/aiyildiz/fitturkai/discussions)
- 📧 **Email**: aiyildiz@gmail.com

---

*Bu dokümantasyon sürekli güncellenmektedir. Son güncellemeler için GitHub repository'sini takip edin.* 