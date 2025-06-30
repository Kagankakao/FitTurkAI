# =====================================================================================
# BÖLÜM 1: KURULUM VE GİRİŞ (A100 GPU İÇİN OPTİMİZE)
# =====================================================================================
!pip install "transformers>=4.38.0" accelerate datasets bitsandbytes peft --quiet

import transformers
import torch
import os
import json
import gc
from huggingface_hub import login
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

print("Transformers kütüphanesi sürümü:", transformers.__version__)

# Hugging Face Hub'a giriş
try:
    # Kendi token'ınızı buraya girin
    login(token="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")
    print("Hugging Face Hub'a başarıyla giriş yapıldı.")
except Exception as e:
    print(f"Hugging Face Hub'a giriş yapılamadı: {e}")

# A100 GPU için ortam optimizasyonları
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# GPU kontrolü ve bellek temizliği
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Toplam GPU belleği: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# =====================================================================================
# BÖLÜM 2: MODEL VE TOKENIZER YÜKLEME (A100 İÇİN OPTİMİZE)
# =====================================================================================

# Türkçe model - A100'de tam precision kullanabiliriz
model_name = "ytu-ce-cosmos/Turkish-Llama-8b-v0.1"

# A100 için daha az agresif quantization (isteğe bağlı)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # A100 bfloat16'yı destekler
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

print("Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model yükleniyor... (8B parametre)")
try:
    # A100'de daha rahat yükleme
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # A100 için optimize
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "35GiB"},  # A100 için güvenli sınır
    )
    print("Model başarıyla yüklendi!")
    
except Exception as e:
    print(f"Quantized model yüklenirken hata: {e}")
    print("Standart model yüklemeye geçiliyor...")
    
    # Fallback: Standart model yükleme
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "35GiB"},
    )
    print("Standart model yüklendi!")

model.config.pad_token_id = tokenizer.pad_token_id

# =====================================================================================
# BÖLÜM 3: PEFT/LoRA KONFIGÜRASYONU
# =====================================================================================
print("LoRA adapterleri hazırlanıyor...")

# Quantized model için hazırlama (eğer quantized ise)
try:
    model = prepare_model_for_kbit_training(model)
    print("Model quantized training için hazırlandı.")
except:
    print("Model zaten standard format'ta.")

# LoRA konfigürasyonu - A100 için daha yüksek rank
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,  # A100'de daha yüksek rank kullanabiliriz
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)

# Parametre sayısını göster
model.print_trainable_parameters()

print(f"Model hazır. GPU bellek kullanımı: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# =====================================================================================
# BÖLÜM 4: GERÇEK VERİ SETİ YÜKLEME VE HAZIRLAMA
# =====================================================================================

def load_and_preprocess_real_data(tokenizer, data_directory="DATA"):
    """Gerçek train.json dosyasını yükle ve işle"""
    
    train_file = os.path.join(data_directory, 'train.json')
    
    # Dosya kontrolü
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"'{train_file}' dosyası bulunamadı! Lütfen dosyanın doğru konumda olduğundan emin olun.")
    
    print(f"Veri seti yükleniyor: {train_file}")
    
    # JSON dosyasını yükle
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Eğer data bir liste değilse, satır satır JSON formatında olabilir
        if isinstance(data, dict):
            # Tek bir JSON objesi ise listeye çevir
            data = [data]
        elif not isinstance(data, list):
            # JSONL formatı kontrolü
            with open(train_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            data = [json.loads(line.strip()) for line in lines if line.strip()]
            
    except json.JSONDecodeError:
        # JSONL formatı dene
        print("Standard JSON formatı okunamadı, JSONL formatı deneniyor...")
        data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Satır {line_num}'de JSON hatası: {e}")
                        continue
    
    print(f"Toplam {len(data)} örnek yüklendi.")
    
    # Veri yapısını analiz et
    if data:
        print("Veri yapısı analizi:")
        first_item = data[0]
        print(f"İlk örnek anahtarları: {list(first_item.keys())}")
        
        # Soru-cevap alanlarını otomatik tespit et
        possible_question_keys = ['soru', 'question', 'input', 'prompt', 'text']
        possible_answer_keys = ['cevap', 'answer', 'output', 'response', 'target']
        
        question_key = None
        answer_key = None
        
        for key in possible_question_keys:
            if key in first_item:
                question_key = key
                break
        
        for key in possible_answer_keys:
            if key in first_item:
                answer_key = key
                break
        
        if not question_key or not answer_key:
            print("UYARI: Standart soru-cevap alanları bulunamadı!")
            print("Mevcut alanlar:", list(first_item.keys()))
            print("Lütfen veri formatınızı kontrol edin.")
            # Kullanıcıdan input al
            print("Hangi alan soru içeriyor?")
            question_key = input("Soru alanı adı: ").strip()
            print("Hangi alan cevap içeriyor?")
            answer_key = input("Cevap alanı adı: ").strip()
        
        print(f"Soru alanı: '{question_key}'")
        print(f"Cevap alanı: '{answer_key}'")
    
    # Dataset oluştur
    dataset = Dataset.from_list(data)
    
    def format_turkish_data(examples):
        """Türkçe soru-cevap formatı"""
        formatted_texts = []
        
        for i in range(len(examples[question_key])):
            try:
                soru = examples[question_key][i]
                cevap = examples[answer_key][i]
                
                # None veya boş değerleri kontrol et
                if not soru or not cevap:
                    continue
                
                text = f"Soru: {soru}\nCevap: {cevap}{tokenizer.eos_token}"
                formatted_texts.append(text)
                
            except (IndexError, KeyError, TypeError) as e:
                print(f"Veri formatı hatası {i}. örnekte: {e}")
                continue
        
        return {"text": formatted_texts}
    
    # Veri formatlaması
    original_columns = dataset.column_names
    dataset = dataset.map(format_turkish_data, batched=True, remove_columns=original_columns)
    
    # Boş örnekleri filtrele
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 10)
    
    print(f"Formatlanmış veri: {len(dataset)} örnek")
    
    # Train-test split
    if len(dataset) > 100:
        test_size = 0.1
    elif len(dataset) > 20:
        test_size = 0.2
    else:
        test_size = 0.0  # Çok az veri varsa split yapma
    
    if test_size > 0:
        split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = dataset.select(range(min(5, len(dataset))))  # En az 5 örnek al
    
    print(f"Eğitim seti: {len(train_dataset)} örnek")
    print(f"Değerlendirme seti: {len(eval_dataset)} örnek")
    
    # Örnek göster
    print("\n--- Örnek Veri ---")
    print(train_dataset[0]['text'][:300] + "...")
    print("------------------\n")
    
    return train_dataset, eval_dataset

# Gerçek veri yükleme
try:
    train_dataset, eval_dataset = load_and_preprocess_real_data(tokenizer)
except Exception as e:
    print(f"Gerçek veri yükleme hatası: {e}")
    print("Lütfen DATA/train.json dosyasının varlığını ve formatını kontrol edin.")
    raise

# =====================================================================================
# BÖLÜM 5: TOKENİZASYON (A100 İÇİN OPTİMİZE)
# =====================================================================================

def tokenize_function(examples):
    """A100 için daha uzun sequence length"""
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=1024,  # A100'de daha uzun kullanabiliriz
        padding=False
    )

print("Tokenizasyon yapılıyor...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("Tokenizasyon tamamlandı.")

# =====================================================================================
# BÖLÜM 6: EĞİTİM YAPILANDIRMASI (A100 İÇİN OPTİMİZE)
# =====================================================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Veri boyutuna göre epoch sayısını ayarla
num_samples = len(tokenized_train)
if num_samples < 100:
    num_epochs = 10
elif num_samples < 1000:
    num_epochs = 5
else:
    num_epochs = 3

print(f"Veri boyutu: {num_samples}, Epoch sayısı: {num_epochs}")

# A100 için optimize edilmiş training arguments
training_args = TrainingArguments(
    output_dir="./turkish-llama-8b-lora-finetuned",
    
    # Batch size - A100'de daha büyük batch kullanabiliriz
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size = 32
    
    # Eğitim parametreleri
    num_train_epochs=num_epochs,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    
    # Precision ve optimizasyon
    bf16=True,  # A100 için bfloat16
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    
    # Kaydetme ve değerlendirme - veri boyutuna göre ayarla
    save_strategy="steps",
    save_steps=max(10, num_samples // 20),  # Veri boyutuna göre ayarla
    eval_strategy="steps",
    eval_steps=max(10, num_samples // 20),
    logging_steps=max(5, num_samples // 40),
    save_total_limit=3,
    
    # Değerlendirme metrikleri
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Diğer optimizasyonlar
    remove_unused_columns=False,
    report_to="tensorboard",
    
    # A100 için ek optimizasyonlar
    max_grad_norm=1.0,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Trainer hazırlandı.")
print(f"Etkili batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Toplam eğitim adımı: ~{len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")

# =====================================================================================
# BÖLÜM 7: EĞİTİMİ BAŞLATMA
# =====================================================================================

print("\n" + "="*60)
print("A100 GPU ÜZERİNDE GERÇEK VERİ İLE EĞİTİM BAŞLIYOR!")
print("="*60)
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Mevcut bellek: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Kullanılabilir bellek: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3:.2f} GB")
print(f"Toplam veri: {len(train_dataset)} eğitim, {len(eval_dataset)} test")
print("="*60)

# Eğitimi başlat
trainer.train(resume_from_checkpoint=True)

print("\nEğitim tamamlandı!")

# =====================================================================================
# BÖLÜM 8: MODEL KAYDETME
# =====================================================================================

final_model_path = "./turkish-llama-8b-lora-final"

# LoRA adapterlerini kaydet
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"Model '{final_model_path}' dizinine kaydedildi.")

# Bellek temizliği
torch.cuda.empty_cache()
gc.collect()

print(f"Final GPU bellek kullanımı: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# =====================================================================================
# BÖLÜM 9: TEST FONKSIYONU
# =====================================================================================

def test_model_with_real_data(model, tokenizer, dataset, num_tests=3):
    """Eğitilen modeli gerçek veri ile test et"""
    
    print("\n" + "="*50)
    print("MODEL GERÇEK VERİ İLE TEST EDİLİYOR")
    print("="*50)
    
    model.eval()
    
    # Rastgele örnekler seç
    import random
    test_indices = random.sample(range(len(dataset)), min(num_tests, len(dataset)))
    
    for i, idx in enumerate(test_indices, 1):
        original_text = dataset[idx]['text']
        
        # Soruyu çıkar
        if "Soru: " in original_text and "Cevap: " in original_text:
            question = original_text.split("Cevap: ")[0].replace("Soru: ", "").strip()
            original_answer = original_text.split("Cevap: ")[1].replace(tokenizer.eos_token, "").strip()
        else:
            continue
            
        print(f"\n{i}. Soru: {question}")
        print(f"Orijinal Cevap: {original_answer[:100]}...")
        
        # Prompt hazırla
        prompt = f"Soru: {question}\nCevap:"
        
        # Tokenize et
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Model Cevabı: {response.strip()}")
        print("-" * 50)

# Modeli gerçek veri ile test et
test_model_with_real_data(model, tokenizer, train_dataset)

print("\n" + "="*60)
print("A100 GPU EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
print("="*60)
print("LoRA adapterleri gerçek veri ile eğitildi ve kaydedildi.")
print("Model artık kullanıma hazır.")
