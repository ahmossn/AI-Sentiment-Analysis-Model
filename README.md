# AI Sentiment Analysis Model 🧠📊

A production-ready NLP application that downloads, stores, and runs a localized **DistilBERT** Transformer network to classify natural text data into positive or negative expressions.

## 🛑 The Problem
Companies process thousands of daily support emails, tweets, and user reviews. Manually combing through feedback to extract brand perception or flag angry customers is slow, expensive, and unscalable.

## ✨ The Solution
This codebase decouples heavy model architecture parameters from application logic. It maps text tokens against pre-trained weight architectures to run localized text classifications in milliseconds.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fetch Large Model Artifacts
GitHub has strict file size caps. Run the ingestion setup to pull the model configurations directly onto your local storage system:
```bash
python train.py
```

### 3. Run Inference Predictions
Execute evaluation text strings against the network:
```bash
python predict.py
```
