import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
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
    dataset = load_dataset('json', data_files=os.path.join(data_directory, 'train.json'), split='train')

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
    Defines the 8B parameter AI model and tokenizer for text generation.
    Includes memory optimizations for large model handling.
    """
    print("Defining the 8B parameter AI model and tokenizer for Causal LM...")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 20:
            print("⚠️  WARNING: Less than 20GB GPU memory detected.")
            print("   Consider using gradient checkpointing and smaller batch sizes.")
        elif gpu_memory >= 24:
            print("✅ Sufficient GPU memory for 8B model fine-tuning.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        print("Pad token not found. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 8B parameter model... This may take a few minutes.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,      # Use half precision to save memory
        device_map="auto",              
        low_cpu_mem_usage=True,         # Reduce CPU memory usage during loading
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model loaded successfully:")
    print(f"  Total parameters: {param_count:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{param_count * 2 / (1024**3):.1f} GB (FP16)")
    
    return model, tokenizer


def fine_tune_ai_model(model, train_dataset, eval_dataset, tokenizer):
    """
    Fine-tunes the 8B parameter generative AI model using optimized hyperparameters.
    Configured specifically for large model fine-tuning with memory optimizations.
    """
    print("Starting the fine-tuning process for 8B parameter Causal LM...")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not Masked LM
    )

    train_dataset_size = len(train_dataset)
    
    batch_size = 2
    gradient_accumulation_steps = 4
    num_epochs = 3
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    total_training_steps = (train_dataset_size / effective_batch_size) * num_epochs
    
    print(f"8B Model Training Configuration:")
    print(f"  Dataset size: {train_dataset_size}")
    print(f"  Per-device batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Total training steps: ~{int(total_training_steps)}")
    print("  Estimated GPU memory needed: ~20-30GB")

    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=50,
        
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        fp16=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.15,
        
        remove_unused_columns=True,
        save_total_limit=2,
        
        max_grad_norm=1.0,
        
        report_to="tensorboard", 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n" + "="*60)
    print("8B MODEL TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Effective Batch Size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print("="*60)

    print("\n⚠️  8B MODEL MEMORY REQUIREMENTS:")
    print("  • Minimum 20GB GPU memory recommended")
    print("  • 24GB+ GPU memory for comfortable training")
    print("  • Monitor GPU memory usage closely")

    trainer.train()
    print("Fine-tuning complete.")
    
    print(f"\nBest model loaded from: {trainer.state.best_model_checkpoint}")
    print(f"Best eval loss: {trainer.state.best_metric}")

    save_directory = "./fine_tuned_FitTurkAI"
    print(f"Saving the best model to {save_directory}...")
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("Best model and tokenizer saved successfully.")

    return trainer


def fitness_ai_assistant_interaction(model, tokenizer, device):
    """
    Handles the interactive conversation with the fine-tuned AI assistant.
    """
    print("\nFitness AI Assistant is ready! Type 'quit' or 'exit' to exit.")
    
    model.eval()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        prompt = f"Soru: {user_input}\nCevap:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
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
    """Main function to run the entire fine-tuning pipeline."""
    device = setup_device()

    model, tokenizer = ai_model_definition()

    train_dataset, eval_dataset = load_and_preprocess_data(tokenizer, data_directory="DATA")

    trainer = fine_tune_ai_model(model, train_dataset, eval_dataset, tokenizer)
    
    best_model = trainer.model
    
    fitness_ai_assistant_interaction(best_model, tokenizer, device)


if __name__ == "__main__":
    main()
