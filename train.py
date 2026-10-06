import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_and_save_model():
    # Utilizing a fast, high-accuracy model for Natural Language Processing (NLP)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    print(f"📥 Downloading '{model_name}' weights from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Save target locally (safely excluded from git tracking via .gitignore)
    output_dir = "./saved_model"
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"✅ Model weights saved successfully to: {output_dir}")

if __name__ == "__main__":
    download_and_save_model()
