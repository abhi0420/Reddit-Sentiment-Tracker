from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_analysis_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
sarcasm_detection_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")

print("Models Loaded Successfully")

def analyze_sentiment(text, aspect=None):
    inputs = tokenizer(text, text_pair=aspect, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0]
    labels = model.config.id2label
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

if __name__ == "__main__":
    text = "I love using transformers for NLP tasks!"
    aspect = "transformers"
    result = analyze_sentiment(text, aspect)
    print(result)