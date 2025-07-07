# 🏥 FitTürkAI - Kişisel Sağlık ve Fitness Asistanı

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-14.1.0-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688)
![React](https://img.shields.io/badge/React-18-61dafb)
![TypeScript](https://img.shields.io/badge/TypeScript-5.8.3-blue)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Türkçe konuşan AI destekli kişisel sağlık ve fitness takip uygulaması**

[🎯 Demo](https://fitturkai-demo.vercel.app) • [📚 Dokümantasyon](#-özellikler) • [🤖 Hugging Face Space](https://huggingface.co/spaces/AIYildiz/AIYildizFitTurkAI) • [📞 Destek](#-iletişim)

</div>

## 📖 Hakkında

FitTürkAI, kullanıcıların sağlık ve fitness hedeflerini takip etmelerine, kişiselleştirilmiş öneriler almalarına ve ilerlemelerini görselleştirmelerine olanak tanıyan modern bir web uygulamasıdır. Türkçe konuşan AI asistanı ile desteklenen uygulama, sağlıklı yaşam yolculuğunuzda size rehberlik eder.

### 🎯 Hedef Kitle
- Sağlıklı yaşam hedefleri olan bireyler
- Fitness ve beslenme takibi yapmak isteyenler
- Kişiselleştirilmiş sağlık önerileri arayan kullanıcılar
- Türkçe destekli AI asistanı tercih edenler

## ✨ Özellikler

### 🤖 AI Asistan
- **Türkçe konuşan AI:** Sağlık ve fitness konularında uzmanlaşmış AI asistanı
- **Kişiselleştirilmiş öneriler:** Kullanıcı verilerine göre özelleştirilmiş tavsiyeler
- **Interaktif chat:** Real-time sohbet deneyimi
- **Sohbet geçmişi:** Tüm konuşmaların kaydedilmesi ve erişimi

### 📊 Takip ve Yönetim
- **Hedef belirleme:** Kilo, fitness, beslenme ve yaşam tarzı hedefleri
- **İlerleme takibi:** Kilometre taşları ve görsel ilerleme çubukları
- **Not alma:** Kategorize edilmiş notlar ve etiketleme sistemi
- **Tarif yönetimi:** Kişisel tariflerin kaydedilmesi ve organizasyonu

### 🎨 Kullanıcı Deneyimi
- **Modern tasarım:** Gradient renkler ve smooth animasyonlar
- **Responsive:** Tüm cihazlarda mükemmel görünüm
- **Dark mode:** Göz yorgunluğunu azaltan karanlık tema
- **Hızlı performans:** localStorage tabanlı hızlı veri erişimi

### 🔐 Güvenlik ve Gizlilik
- **Yerel veri saklama:** Veriler kullanıcının tarayıcısında güvenli şekilde saklanır
- **Basit authentication:** E-posta tabanlı güvenli giriş sistemi
- **Veri kontrolü:** Kullanıcının veriler üzerinde tam kontrolü

## 🛠️ Teknoloji Stack'i

### Frontend
- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript 5.8.3
- **UI Library:** React 18
- **Styling:** Tailwind CSS 3.4.17
- **Animations:** Framer Motion 9.1.7
- **Icons:** Heroicons 2.2.0
- **Charts:** Chart.js, Recharts, Ant Design Plots

### Backend
- **Framework:** FastAPI 0.104.1
- **Language:** Python 3.9+
- **AI Integration:** Hugging Face Spaces API
- **HTTP Client:** Gradio Client
- **CORS:** Cross-origin resource sharing

### AI Model
- **Platform:** Hugging Face Spaces
- **Model:** AIYildiz/AIYildizFitTurkAI
- **Language:** Türkçe optimized
- **Specialization:** Sağlık ve fitness danışmanlığı

## 🚀 Kurulum ve Çalıştırma

### Seçenek 1: Hugging Face Space API (Önerilen - Ücretsiz)

Bu seçenek ile AI modelini kendi bilgisayarınızda çalıştırmanıza gerek yok. Ücretsiz Hugging Face API kullanılır.

#### 1. Projeyi Klonlayın
```bash
git clone https://github.com/aiyildiz/fitturkai.git
cd fitturkai
```

#### 2. Frontend Kurulumu
```bash
npm install
npm run dev
```

#### 3. Backend Kurulumu
```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### 4. Uygulamayı Açın
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

### Seçenek 2: Yerel Model (Gelişmiş Kullanıcılar)

Bu seçenek ile AI modelini kendi bilgisayarınızda çalıştırabilirsiniz.

#### 1. Model İndirme
AI modelini [Hugging Face](https://huggingface.co/AIYildiz/AIYildizFitTurkAI) üzerinden indirin ve `backend/` klasörüne yerleştirin.

#### 2. Backend Konfigürasyonu
`backend/main.py` dosyasında yerel model kullanımı için gerekli değişiklikleri yapın:

```python
# Yerel model kullanımı için
from llama_cpp import Llama

llm = Llama(
    model_path="./model-dosyasi.gguf",
    n_ctx=4096,
    n_threads=8,
    verbose=False
)
```

#### 3. Sistem Gereksinimleri
- **RAM:** Minimum 8GB (16GB önerilen)
- **Storage:** 5-10GB boş alan
- **CPU:** Modern işlemci (GPU opsiyonel)

## 📁 Proje Yapısı

```
fitturkai/
├── 📁 src/                           # Frontend kaynak kodları
│   ├── 📁 app/                      # Next.js App Router sayfaları
│   │   ├── 📁 auth/                 # Authentication sayfaları
│   │   ├── 📁 chat/                 # Chat sayfası
│   │   ├── 📁 goals/                # Hedefler sayfası
│   │   ├── 📁 notes/                # Notlar sayfası
│   │   ├── 📁 profile/              # Profil sayfası
│   │   ├── 📁 recipes/              # Tarifler sayfası
│   │   └── 📄 layout.tsx            # Ana layout
│   ├── 📁 components/               # React bileşenleri
│   │   ├── 📄 Sidebar.tsx           # Yan menü
│   │   └── 📄 Dashboard.tsx         # Dashboard widget'ı
│   ├── 📁 data/                     # JSON veri dosyaları
│   │   ├── 📄 chats.json           # Örnek sohbet verileri
│   │   ├── 📄 goals.json           # Varsayılan hedefler
│   │   ├── 📄 notes.json           # Örnek notlar
│   │   └── 📄 recipes.json         # Tarif koleksiyonu
│   └── 📁 utils/                    # Yardımcı fonksiyonlar
│       └── 📁 api/                  # API istemci fonksiyonları
│           └── 📄 ai-assistant.ts   # AI asistan entegrasyonu
├── 📁 backend/                      # Backend API
│   ├── 📄 main.py                  # FastAPI uygulaması
│   ├── 📄 requirements.txt         # Python bağımlılıkları
│   └── 📄 README.md                # Backend dokümantasyonu
├── 📁 public/                       # Statik dosyalar
├── 📄 package.json                 # NPM bağımlılıkları
├── 📄 tailwind.config.js           # Tailwind CSS konfigürasyonu
├── 📄 next.config.mjs              # Next.js konfigürasyonu
└── 📄 README.md                    # Bu dosya
```

## 🔧 Konfigürasyon

### Environment Variables
```bash
# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Backend
HUGGING_FACE_TOKEN=your_token_here  # Opsiyonel
```

### Hugging Face Space Değiştirme
Backend'de farklı bir Space kullanmak için `backend/main.py` dosyasında:

```python
# Mevcut Space
client = Client("AIYildiz/AIYildizFitTurkAI")

# Yeni Space ile değiştir
client = Client("your-username/your-space-name")
```

## 🎮 Kullanım

### 1. Hesap Oluşturma
- E-posta adresi ile kayıt olun
- Demo hesap: `fitturkai@demo.com` / `123456`

### 2. Profil Ayarlama
- Kişisel bilgilerinizi girin
- Sağlık hedeflerinizi belirleyin
- Tercihleri ayarlayın

### 3. AI Asistan ile Sohbet
- Chat sayfasında AI asistanı ile konuşun
- Sağlık ve fitness sorularınızı sorun
- Kişiselleştirilmiş öneriler alın

### 4. Hedefler Belirleme
- Kilo, fitness, beslenme hedefleri ekleyin
- Kilometre taşları oluşturun
- İlerlemenizi takip edin

### 5. Notlar ve Tarifler
- Önemli notlarınızı kaydedin
- Favori tariflerinizi saklayın
- Kategorilere ayırın ve etiketleyin

## 🧪 Test ve Geliştirme

### Frontend Geliştirme
```bash
npm run dev          # Geliştirme sunucusu
npm run build        # Production build
npm run lint         # ESLint kontrolü
npm run lint:fix     # ESLint otomatik düzeltme
npm run format       # Prettier formatlaması
```

### Backend Test
```bash
cd backend
python main.py       # Sunucuyu başlat
# Test için: http://localhost:8000/docs
```

### API Endpoints
- `POST /chat` - AI ile sohbet
- `GET /health` - Sistem durumu kontrolü

## 🌟 Özellik Roadmap

### Yakın Gelecek (v1.1)
- [ ] Haftalık/aylık raporlar
- [ ] Egzersiz video entegrasyonu
- [ ] Besin değeri hesaplayıcısı
- [ ] AI Agent sistemi ile otomatik haftalık öğün planı 
- [ ] 

### Orta Vadeli (v1.5)
- [ ] Mobil uygulama (React Native)
- [ ] Wearable device entegrasyonu
- [ ] Gelişmiş analytics dashboard
- [ ] Multi-language support

### Uzun Vadeli (v2.0)
- [ ] Machine learning insights
- [ ] Doktor/diyetisyen bağlantısı
- [ ] Community features
- [ ] Premium subscription

## 🤝 Katkıda Bulunma

### Katkı Türleri
- 🐛 Bug raporları
- 💡 Özellik önerileri
- 📝 Dokümantasyon iyileştirmeleri
- 🔧 Kod katkıları
- 🎨 UI/UX tasarım önerileri

### Katkı Süreci
1. **Fork:** Bu repository'yi fork edin
2. **Branch:** Yeni bir feature branch oluşturun
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit:** Değişikliklerinizi commit edin
   ```bash
   git commit -m 'feat: Add amazing feature'
   ```
4. **Push:** Branch'inizi push edin
   ```bash
   git push origin feature/amazing-feature
   ```
5. **PR:** Pull Request açın

### Commit Mesaj Formatı
```
type(scope): description

Types: feat, fix, docs, style, refactor, test, chore
```

## 🚀 Production Deployment Rehberi

### ☁️ Vercel (Frontend) + Railway (Backend)

**Frontend Vercel Deployment:**

1. **🔗 GitHub Repository Bağlayın:**
```bash
# Repository'yi Vercel'e bağlayın
https://vercel.com/new

# Root directory: FitTürkAI Web Demo
# Framework: Next.js
# Node.js Version: 18.x
```

2. **⚙️ Environment Variables (Vercel):**
```env
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
NODE_ENV=production
```

3. **🔧 Build Settings:**
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "installCommand": "npm ci",
  "devCommand": "npm run dev"
}
```

**Backend Railway Deployment:**

1. **🚂 Railway Setup:**
```bash
# Railway CLI kurulumu
npm install -g @railway/cli

# Login ve deployment
railway login
railway init
railway up
```

2. **⚙️ Environment Variables (Railway):**
```env
HUGGING_FACE_SPACE=AIYildiz/AIYildizFitTurkAI
HUGGING_FACE_TOKEN=your_token_here
CORS_ORIGINS=["https://your-frontend.vercel.app"]
API_HOST=0.0.0.0
PORT=8000
PYTHONPATH=/app
```

3. **📁 Railway Procfile:**
```bash
# Procfile
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```



### 🌐 AWS/DigitalOcean Deployment

**1. AWS EC2 Setup:**

```bash
# EC2 instance kurulumu
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx nodejs npm python3-pip

# Node.js ve Python kurulumu
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# SSL sertifikası
sudo certbot --nginx -d yourdomain.com -d api.yourdomain.com
```

**2. Nginx Reverse Proxy:**

```nginx
# /etc/nginx/sites-available/fitturkai
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 📊 Production Monitoring

**1. Health Checks:**

```bash
# Automated health check script
#!/bin/bash
# health-check.sh

FRONTEND_URL="https://yourdomain.com"
BACKEND_URL="https://api.yourdomain.com/health"

# Frontend check
if curl -f $FRONTEND_URL > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend is down"
    # Send alert (Slack, email, etc.)
fi

# Backend check
if curl -f $BACKEND_URL > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend is down" 
    # Send alert
fi
```

**2. Log Management:**

```bash
# Application logs
tail -f /var/log/nginx/access.log | jq '.'

# Error monitoring
grep "ERROR" /var/log/fitturkai/*.log

# Real-time log monitoring
journalctl -f -u fitturkai
```

**3. Performance Monitoring:**

```bash
# Resource usage
htop
ps aux | grep fitturkai

# Application metrics
curl https://api.yourdomain.com/metrics

# Uptime monitoring
systemctl status fitturkai
```

### 🔒 Production Security

**1. Environment Security:**

```env
# .env.production
NODE_ENV=production
NEXT_PUBLIC_API_URL=https://api.yourdomain.com

# Güvenlik headers
SECURITY_HEADERS=true
CSP_ENABLED=true
HSTS_ENABLED=true
```

**2. Rate Limiting:**

```python
# backend/main.py production config
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("20/minute")  # Production rate limit
async def chat_endpoint(request: Request, ...):
    pass
```

**3. CORS Production Settings:**

```python
# Strict CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Only your domain
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
```

### 🚀 CI/CD Pipeline

**GitHub Actions Workflow:**

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: 18
          
      - name: Install dependencies
        run: npm ci
        
      - name: Run tests
        run: npm test
        
      - name: Build application
        run: npm run build
        
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'
```

---

## ⚠️ Önemli Uyarı ve Sorumluluk Reddi

### 🏥 Tıbbi Sorumluluk Reddi

**FitTürkAI bir yapay zeka asistanıdır ve profesyonel tıbbi tavsiye, tanı veya tedavi sağlamaz.**

- **❗ Tıbbi Acil Durumlar**: Acil sağlık durumlarında derhal 112'yi arayın
- **👩‍⚕️ Profesyonel Danışmanlık**: Sağlık kararları alırken mutlaka doktor, diyetisyen veya sağlık uzmanına danışın
- **🔬 Bilimsel Amaç**: Bu sistem sadece genel bilgilendirme ve eğitim amaçlıdır
- **🚫 Sorumluluk**: FitTürkAI'nin verdiği bilgilere dayanılarak alınan kararlardan geliştiriciler sorumlu değildir
- **📋 Kişisel Durumlar**: Her bireyin sağlık durumu farklıdır, kişiselleştirilmiş planlar için uzman desteği alın

### 🔒 Gizlilik ve Veri Güvenliği

- **💾 Yerel Depolama**: Verileriniz sadece tarayıcınızda saklanır
- **🚫 Sunucu Kayıtları**: Kişisel sağlık bilgileriniz sunucularımızda saklanmaz
- **🔐 Güvenlik**: Hassas bilgilerinizi paylaşırken dikkatli olun
- **🗑️ Veri Silme**: Tarayıcı verilerini istediğiniz zaman silebilirsiniz

---

## 📝 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📞 İletişim

### Geliştirici
- **GitHub:** [@aiyildiz](https://github.com/aiyildiz)
- **E-posta:** aiyildiz@gmail.com

### Destek
- **Issues:** [GitHub Issues](https://github.com/aiyildiz/fitturkai/issues)
- **Discussions:** [GitHub Discussions](https://github.com/aiyildiz/fitturkai/discussions)
- **Documentation:** [Wiki](https://github.com/aiyildiz/fitturkai/wiki)

### AI Model
- **Hugging Face Space:** [AIYildiz/AIYildizFitTurkAI](https://huggingface.co/spaces/AIYildiz/AIYildizFitTurkAI)
- **Model Repository:** [Model Detayları](https://huggingface.co/AIYildiz/AIYildizFitTurkAI)
- **Q8 Model Repository:** [Model Detayları](https://huggingface.co/AIYildiz/AIYildiz-FitTurkAI-Q8)
## 🙏 Teşekkürler

Bu projeye katkıda bulunan herkese teşekkür ederiz:

- **Hugging Face** - AI model hosting için
- **Next.js Team** - Framework desteği için
- **FastAPI Team** - Backend framework için
- **Tailwind CSS** - UI styling için
- **Open Source Community** - Kullanılan tüm paketler için

## 📊 İstatistikler

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/aiyildiz/fitturkai?style=social)
![GitHub forks](https://img.shields.io/github/forks/aiyildiz/fitturkai?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/aiyildiz/fitturkai?style=social)
![GitHub issues](https://img.shields.io/github/issues/aiyildiz/fitturkai)
![GitHub pull requests](https://img.shields.io/github/issues-pr/aiyildiz/fitturkai)

</div>

---

<div align="center">

**🏥 FitTürkAI ile sağlıklı yaşam yolculuğunuza başlayın!**

Made with ❤️ in Turkey 🇹🇷

</div>
