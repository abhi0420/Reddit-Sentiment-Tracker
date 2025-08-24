from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else -1

ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL_NAME)

sentiment_analysis_pipeline = pipeline("sentiment-analysis",model="cardiffnlp/twitter-roberta-base-sentiment-latest")
sarcasm_detection_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")

print("Models Loaded Successfully")

def signed_sent_score(label_probs : dict[str, float]) -> float:
    label_probs = {k.lower() : v for k,v in label_probs.items()}
    pos = label_probs.get("positive", 0)
    neg = label_probs.get("negative", 0)
    neu = label_probs.get("neutral", 0)
    return float(max(-1.0, min(pos - neg, 1.0)))    

def label_from_signed(score : float) -> dict[str, float]:
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    
    return "neutral"

@torch.inference_mode()
def analyze_sentiment(text, title, aspect=None):
    if not title:
        title = ""
    full_text = (title + "\n" + text).strip().lower()
    if aspect and aspect.lower() in full_text.lower():
        #print(f"Aspect '{aspect}' found in text or title.")
        inputs = tokenizer(text, text_pair=aspect, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs =  F.softmax(outputs.logits, dim=-1)[0].cpu().tolist()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        label_probs = {id2label[i].lower(): probs[i] for i in range(len(probs))}
        score = signed_sent_score(label_probs)
        label = label_from_signed(score)
        method = "absa"
     
    else:
        result = sentiment_analysis_pipeline(text)[0]
        top_label = result['label'].lower()
        top_prob = float(result['score'])
        # naive distribution: assign leftover to neutral
        if top_label == "positive":
            label_probs = {"positive": top_prob, "negative": 1 - top_prob - 1e-6, "neutral": 1e-6}
        elif top_label == "negative":
            label_probs = {"negative": top_prob, "positive": 1 - top_prob - 1e-6, "neutral": 1e-6}
        else:
            label_probs = {"neutral": top_prob, "positive": (1 - top_prob) / 2, "negative": (1 - top_prob) / 2}

        score = signed_sent_score(label_probs)
        label = label_from_signed(score)    
        method = "generic"

    check_sarcasm, sarcasm_score = analyze_sarcasm(text)
    if check_sarcasm and abs(sarcasm_score) > 0.5  :
        score = -0.75 * score
        label = label_from_signed(score)
    return {"label": label, "score": score, "method": method}

@torch.inference_mode()
def analyze_sarcasm(text):
    sarcasm = sarcasm_detection_pipeline(text)[0] 
    if sarcasm['label'] == 'irony':
        if sarcasm['score'] > 0.8:
            return True, sarcasm['score']
    return False, sarcasm['score']

if __name__ == "__main__":
    text = "X is really great at making terrible decisions"
    aspect = "X"
    result = analyze_sentiment(text, "", aspect)
    print(result)