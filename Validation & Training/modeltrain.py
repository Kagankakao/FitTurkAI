import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
import json

def setup_device():
    """Sets up the device for training (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_and_preprocess_data(tokenizer, data_directory="DATA"):
    """
    Loads and preprocesses data from a directory of JSON files.
    This function is configured for your data structure.

    Args:
        tokenizer: The tokenizer to use.
        data_directory (str): The path to the folder containing JSON files.

    Returns:
        tuple: A tuple containing the processed train and evaluation datasets.
    """
    print("Loading and preprocessing data...")
    # Create a dummy JSON file for demonstration if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    train_file_path = os.path.join(data_directory, 'train.json')
    if not os.path.exists(train_file_path):
        print(f"Creating dummy data at {train_file_path}")
        dummy_data = [{"soru": f"Soru {i}", "cevap": f"Cevap {i}"} for i in range(100)]
        with open(train_file_path, 'w', encoding='utf-8') as f:
            for item in dummy_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dataset = load_dataset('json', data_files=train_file_path, split='train')

    def format_and_tokenize(examples):
        """
        Formats your 'soru' and 'cevap' data and tokenizes it.
        """
        questions = [f"Soru: {q}" for q in examples["soru"]]
        answers = [f"Cevap: {a}{tokenizer.eos_token}" for a in examples["cevap"]]

        full_texts = [q + "\n" + a for q, a in zip(questions, answers)]

        return tokenizer(full_texts, padding="max_length", truncation=True, max_length=256)

    tokenized_datasets = dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )

    split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Data loading and preprocessing complete. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def ai_model_definition(model_name="ytu-ce-cosmos/Turkish-Llama-8b-v0.1"):
    """
    Defines the AI model and tokenizer for text generation using QLoRA.
    Uses 4-bit quantization and LoRA for memory-efficient fine-tuning.
    """
    print("Defining the AI model and tokenizer with QLoRA...")

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available GPU memory: {gpu_memory:.1f} GB")
        print("✅ Using QLoRA for memory-efficient fine-tuning.")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        print("Pad token not found. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with 4-bit quantization... This may take a few minutes.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Print model info
    model.print_trainable_parameters()

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("QLoRA Model loaded successfully:")
    print(f"  Total parameters: {param_count:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / param_count:.2f}%")

    return model, tokenizer


def fine_tune_ai_model(model, train_dataset, eval_dataset, tokenizer):
    """
    Fine-tunes the generative AI model using QLoRA.
    Much more memory efficient than full fine-tuning.
    """
    print("Starting the QLoRA fine-tuning process...")

    model.config.use_cache = False

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not Masked LM
    )

    train_dataset_size = len(train_dataset)

    # QLoRA allows for larger batch sizes due to memory efficiency
    batch_size = 4  # Can be larger with QLoRA
    gradient_accumulation_steps = 4
    num_epochs = 3  # Can train for more epochs with QLoRA
    effective_batch_size = batch_size * gradient_accumulation_steps
    total_training_steps = (train_dataset_size / effective_batch_size) * num_epochs

    print(f"QLoRA Training Configuration:")
    print(f"  Dataset size: {train_dataset_size}")
    print(f"  Per-device batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Total training steps: ~{int(total_training_steps)}")

    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=10,

        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,  # Can be larger for eval
        gradient_accumulation_steps=gradient_accumulation_steps,

        # Use BF16 for stability with QLoRA
        bf16=True,
        fp16=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,

        learning_rate=2e-4,  # Higher learning rate for LoRA
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.001,  # Lower weight decay for LoRA

        remove_unused_columns=True,
        save_total_limit=2,
        max_grad_norm=0.3,  # Lower for LoRA

        report_to="tensorboard",

        # QLoRA specific optimizations
        optim="paged_adamw_32bit",  # Memory-efficient optimizer
        save_only_model=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    import warnings
    warnings.filterwarnings("ignore", message="`tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`.")

    trainer.train()
    print("QLoRA fine-tuning complete.")

    if trainer.state.best_model_checkpoint:
        print(f"\nBest model loaded from: {trainer.state.best_model_checkpoint}")
        print(f"Best eval loss: {trainer.state.best_metric}")
    else:
        print("\nTraining finished, but no best model checkpoint was found.")

    save_directory = "./fine_tuned_FitTurkAI_QLoRA"
    print(f"Saving the LoRA adapters to {save_directory}...")

    # Save only the LoRA adapters (much smaller)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("LoRA adapters and tokenizer saved successfully.")

    return trainer


def fitness_ai_assistant_interaction(model, tokenizer, device):
    """
    Handles the interactive conversation with the fine-tuned AI assistant.
    """
    instruction = """Sen, FitTürkAI adında, Bütünsel, Empatik ve Proaktif bir Sağlıklı Yaşam Koçusun. Görevin, kullanıcılara sadece beslenme ve egzersiz planları sunmak değil, aynı zamanda uyku, stres yönetimi gibi bütünsel sağlık konularında da rehberlik etmektir.

ÖNEMLI KURALLAR:
- Doktor olmadığını, tıbbi tavsiye vermediğini belirt
- YASAKLI KELIMELER: "Tedavi", "reçete", "diyet listesi", "zayıflama programı", "garanti"
- İZIN VERILEN KELIMELER: "Rehber", "çerçeve", "yol haritası", "öneri", "fikir"
- Empatik, cesaret verici bir dil kullan
- Porsiyon ve kalori bilgilerini sadece "yaklaşık", "tahmini" ifadeleriyle ver

Kullanıcının sorusunu bu role göre yanıtla:"""

    print("\nFitness AI Assistant is ready! Type 'quit' or 'exit' to exit.")

    model.eval()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        prompt = f"{instruction}\n\nSoru: {user_input}\nCevap:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
            )

        response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        print(f"AI Assistant: {response_text.strip()}")


def main():
    """Main function to run the entire QLoRA fine-tuning pipeline."""
    device = setup_device()

    model, tokenizer = ai_model_definition()

    train_dataset, eval_dataset = load_and_preprocess_data(tokenizer)

    trainer = fine_tune_ai_model(model, train_dataset, eval_dataset, tokenizer)

    best_model = trainer.model

    fitness_ai_assistant_interaction(best_model, tokenizer, device)


if __name__ == "__main__":
    main()
