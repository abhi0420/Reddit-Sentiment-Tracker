from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

def load_models():

    DEVICE = "cuda" if torch.cuda.is_available() else -1

    ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
    absa_model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL_NAME)

    sentiment_analysis_pipeline = pipeline("sentiment-analysis",model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    sarcasm_detection_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")

    print("Models Loaded Successfully")

    tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL_NAME)

    return {
        "absa_model": absa_model,
        "sentiment_analysis_pipeline": sentiment_analysis_pipeline,
        "sarcasm_detection_pipeline": sarcasm_detection_pipeline,
        "tokenizer": tokenizer
    }


@torch.inference_mode()
def analyze_sentiment(model, sentiment_analysis_pipeline, sarcasm_detection_pipeline, tokenizer, text, title, aspect=None):
    if not title:
        title = ""
    full_text = (title + "\n" + text).strip()
    
    if aspect and aspect.lower() in full_text.lower():
        print("Performing ABSA")
        # Truncation logic (keep this part)
        tokens = tokenizer.encode(full_text, truncation=False)
        if len(tokens) > tokenizer.model_max_length:
            aspect_index = full_text.lower().index(aspect.lower())
            start_index = max(0, aspect_index - tokenizer.model_max_length // 2)
            end_index = start_index + tokenizer.model_max_length
            full_text = full_text[start_index:end_index]
        
        inputs = tokenizer(full_text, text_pair=aspect, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].cpu().tolist()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        label_probs = {id2label[i].lower(): probs[i] for i in range(len(probs))}
        print(label_probs)
        # 🎯 SIMPLIFIED: Just use the max probability
        label = max(label_probs.items(), key=lambda x: x[1])[0]
        pos = label_probs.get("positive", 0)
        neg = label_probs.get("negative", 0)
        confidence = max(label_probs.values())
        score = confidence if label == "positive" else (-confidence if label == "negative" else pos-neg)
        method = "absa"
     
    else:
        # Generic sentiment - even simpler!
        if len(text) > 500:
            text = text[:500]
            
        result = sentiment_analysis_pipeline(text)[0]
        label = result['label'].lower()
        confidence = result['score']
        score = confidence if label == "positive" else (-confidence if label == "negative" else 0.0)
        method = "generic"

    # Sarcasm detection (keep this logic - it's valuable)
    try:
        if len(text) > 500:
            text = text[:500]
        check_sarcasm, sarcasm_score = analyze_sarcasm(text, sarcasm_detection_pipeline)
    except Exception as e:
        check_sarcasm, sarcasm_score = False, 0.0
    
    if check_sarcasm and abs(sarcasm_score) > 0.5:
        # Flip sentiment if sarcastic
        if label == "positive":
            label = "negative"
            score = -abs(score)
        elif label == "negative":
            label = "positive"
            score = abs(score)
    
    return {
        "label": label, 
        "score": score, 
        "confidence": confidence,
        "method": method,
        "sarcasm_detected": check_sarcasm
    }

@torch.inference_mode()
def analyze_sarcasm(text, sarcasm_detection_pipeline):
    sarcasm = sarcasm_detection_pipeline(text)[0] 
    if sarcasm['label'] == 'irony':
        if sarcasm['score'] > 0.8:
            return True, sarcasm['score']
    return False, sarcasm['score']

if __name__ == "__main__":
    nlp_models = load_models()
    absa_model = nlp_models.get("absa_model")
    sentiment_analysis_pipeline = nlp_models.get("sentiment_analysis_pipeline")
    sarcasm_detection_pipeline = nlp_models.get("sarcasm_detection_pipeline")
    tokenizer = nlp_models.get("tokenizer")

    text = "Hear me out. This might sound extreme, but I genuinely believe itâ€™s time to consider abandoning Delhi.relocating Delhiâ€™s population across smaller cities could help reduce strain on resources and make a dent in urban pollution nationwide."
    aspect = "Delhi"
    result = analyze_sentiment(absa_model, sentiment_analysis_pipeline, sarcasm_detection_pipeline, tokenizer, text, "HOT TAKE : Delhi Should be Abandoned.", aspect)
    print(result)