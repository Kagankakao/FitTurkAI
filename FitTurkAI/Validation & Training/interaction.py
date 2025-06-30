def test_model_with_custom_questions(model, tokenizer):
    """Eğitilen modeli özel sorularla test et"""
    
    print("\n" + "="*60)
    print("MODEL ÖZEL SORULARLA TEST EDİLİYOR")
    print("="*60)
    
    model.eval()
    
    # Test soruları listesi - istediğiniz gibi değiştirebilirsiniz
    questions = [
        "179 boyunda 120 kilo erkeğim 30 kilo vermek istiyorum bunun için  ne yapmalıyım ne yemeliyim bana haftalık diyet listesi yap",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Soru: {question}")
        
        # Prompt hazırla
        prompt = f"Soru: {question}\nCevap:"
        
        # Tokenize et
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate cevap
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
        
        # Decode cevap
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Model Cevabı: {response.strip()}")
        print("-" * 50)

def interactive_test_model(model, tokenizer):
    """İnteraktif test - kullanıcının girdiği sorularla test et"""
    
    print("\n" + "="*60)
    print("İNTERAKTİF MODEL TEST MODU")
    print("Çıkmak için 'quit' yazın")
    print("="*60)
    
    model.eval()
    
    while True:
        question = input("\nSorunuzu yazın: ").strip()
        
        if question.lower() in ['quit', 'exit', 'çık', 'q']:
            print("Test modu sonlandırıldı.")
            break
        
        if not question:
            print("Lütfen bir soru yazın.")
            continue
        
        print(f"\nSoru: {question}")
        
        # Prompt hazırla
        prompt = f"Soru: {question}\nCevap:"
        
        # Tokenize et
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate cevap
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
        
        # Decode cevap
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Model Cevabı: {response.strip()}")
        print("-" * 50)

# Önceden tanımlanmış sorularla test
test_model_with_custom_questions(model, tokenizer)

# İnteraktif test için (isteğe bağlı)
# interactive_test_model(model, tokenizer)
