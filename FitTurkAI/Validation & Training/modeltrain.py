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

def create_dummy_data_for_testing():
    """
    Creates a dummy DATA folder and a sample JSON file for quick testing.
    This now uses 'soru' and 'cevap' to match your data structure.
    """
    print("Creating dummy data for demonstration...")
    data_dir = "DATA"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Dummy data now matches the Turkish key names
    dummy_data = [
        {
            "soru": "Kişiye özel diyet planlaması neden önemlidir?",
            "cevap": "Kişiye özel diyet planları, bireyin yaşam tarzına, metabolizmasına ve hedeflerine uygun olduğu için daha sürdürülebilir ve etkilidir."
        },
        {
            "soru": "Sağlıklı bir diyetin temel prensipleri nelerdir?",
            "cevap": "Sağlıklı bir diyetin temel prensipleri dengeli makro besin alımı, bol lifli gıda tüketimi ve işlenmiş gıdalardan kaçınmaktır."
        }
    ]

    # Use your actual filename for consistency
    with open(os.path.join(data_dir, "diyetVeEgzersiz.json"), "w", encoding="utf-8") as f:
        for item in dummy_data:
            # Writing as JSON Lines format
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Dummy data created in 'DATA/diyetVeEgzersiz.json'")


def setup_device():
    """Sets up the device for training (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_and_preprocess_data(tokenizer, data_directory="DATA"):
    """
    Loads and preprocesses data from a directory of JSON files.
    This function is now correctly configured for your data.

    Args:
        tokenizer: The tokenizer to use.
        data_directory (str): The path to the folder containing JSON files.

    Returns:
        tuple: A tuple containing the processed train and evaluation datasets.
    """
    print("Loading and preprocessing data...")
    # This will load all .json files in the specified directory
    dataset = load_dataset('json', data_dir=data_directory, split='train')

    def format_and_tokenize(examples):
        """
        Formats your 'soru' and 'cevap' data and tokenizes it.
        Uses Turkish prompts for better performance with your data.
        """
        # **This is the corrected part**
        questions = [f"Soru: {q}" for q in examples["soru"]]
        answers = [f"Cevap: {a}{tokenizer.eos_token}" for a in examples["cevap"]]

        full_texts = [q + "\n" + a for q, a in zip(questions, answers)]

        return tokenizer(full_texts, padding="max_length", truncation=True, max_length=256)

    tokenized_datasets = dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Split the dataset into training and evaluation sets
    split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Data loading and preprocessing complete. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def ai_model_definition():
    """
    Defines the AI model and tokenizer for text generation.
    """
    print("Defining the AI model and tokenizer for Causal LM...")
    model_name = "ytu-ce-cosmos/Turkish-Llama-8b-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Model and tokenizer definition complete.")
    return model, tokenizer


def fine_tune_ai_model(model, train_dataset, eval_dataset, tokenizer):
    """
    Fine-tunes the generative AI model using the Trainer API.
    """
    print("Starting the fine-tuning process for Causal LM...")

    # Use the appropriate data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not Masked LM
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("Fine-tuning complete.")


def save_fine_tuned_ai(model, tokenizer, save_directory="./fine_tuned_fitness_ai"):
    """Saves the fine-tuned model and tokenizer."""
    print(f"Saving the fine-tuned model to {save_directory}...")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("Model and tokenizer saved successfully.")


def load_fine_tuned_ai(model_directory="./fine_tuned_fitness_ai"):
    """Loads a fine-tuned generative model and tokenizer."""
    print(f"Loading the fine-tuned model from {model_directory}...")
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory)
    print("Model loaded successfully.")
    return model, tokenizer


def fitness_ai_assistant_interaction(model, tokenizer, device):
    """
    Handles the interactive conversation with the fine-tuned AI assistant.
    """
    print("\nFitness AI Assistant is ready! Type 'quit' to exit.")
    model.to(device)
    model.eval()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # Format the user's input just like the training data
        prompt = f"Soru: {user_input}\nCevap:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up the output to only show the generated answer
        try:
            answer = response_text.split("Cevap:")[1].strip()
        except IndexError:
            answer = "I'm not sure how to answer that yet."

        print(f"AI Assistant: {answer}")


def main():
    """Main function to run the entire fine-tuning pipeline."""
    # This function creates a dummy 'diyetVeEgzersiz.json' file if you don't have one.
    # You can comment this out if you are using your real file.
    if not os.path.exists("DATA"):
        create_dummy_data_for_testing()

    device = setup_device()

    model, tokenizer = ai_model_definition()
    model.to(device)

    train_dataset, eval_dataset = load_and_preprocess_data(tokenizer, data_directory="DATA")

    fine_tune_ai_model(model, train_dataset, eval_dataset, tokenizer)

    save_fine_tuned_ai(model, tokenizer)

    # Load the best model saved during training and interact with it
    loaded_model, loaded_tokenizer = load_fine_tuned_ai(model_directory="./fine_tuned_fitness_ai")
    fitness_ai_assistant_interaction(loaded_model, loaded_tokenizer, device)


if __name__ == "__main__":
    main()