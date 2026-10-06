import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_sentiment(text):
    model_dir = "./saved_model"
    
    if not os.path.exists(model_dir):
        print("❌ Error: Local model files not found. Please run 'python train.py' first.")
        return

    # Load model binaries locally
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Tokenize input text to tensor vectors
    inputs = tokenizer(text, return_tensors="pt")
    
    # Execute forward pass inference without calculating gradients
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = torch.argmax(logits, dim=1).item()
    
    # Map predictions back to semantic labels
    labels = ["Negative", "Positive"]
    return labels[predicted_class_id]

if __name__ == "__main__":
    sample_text = "I absolutely love building AI projects and sharing them on GitHub!"
    print(f"📝 Raw Text Input: '{sample_text}'")
    result = predict_sentiment(sample_text)
    print(f"🤖 AI Prediction Sentiment: {result}")
