from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

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
    max_score = max(labels, key = labels.get)
    check_sarcasm = analyze_sarcasm(text)
    if check_sarcasm:
        if labels[max_score].lower() ==  "positive":
            labels[max_score] = "negative"
        else:
            labels[max_score] = "positive"
    return {labels[max_score]: float(probs[max_score])}

def analyze_sarcasm(text):
    sarcasm = sarcasm_detection_pipeline(text)[0] 
    if sarcasm['label'] == 'irony':
        if sarcasm['score'] > 0.8:
            return True
    return False

if __name__ == "__main__":
    text = "You are great at making terrible decisions"
    aspect = "transformers"
    result = analyze_sentiment(text, aspect)
    print(result)