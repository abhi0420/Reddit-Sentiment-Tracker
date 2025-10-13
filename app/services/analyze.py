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
def analyze_sentiment(model,sentiment_analysis_pipeline, sarcasm_detection_pipeline, tokenizer, text, title, aspect=None):
    #print("Aspect:", aspect)
    if not title:
        title = ""
    full_text = (title + "\n" + text).strip()
    
    if aspect and aspect.lower() in full_text.lower():
        print("Performing ABSA")
        # Use the correct tokenizer for length checking
        tokens = tokenizer.encode(full_text, truncation=False)
        if len(tokens) > tokenizer.model_max_length:
            aspect_index = full_text.lower().index(aspect.lower())
            start_index = max(0, aspect_index - tokenizer.model_max_length // 2)
            end_index = start_index + tokenizer.model_max_length
            full_text = full_text[start_index:end_index]
        
        inputs = tokenizer(full_text, text_pair=aspect, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        print(outputs)
        probs =  F.softmax(outputs.logits, dim=-1)[0].cpu().tolist()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        label_probs = {id2label[i].lower(): probs[i] for i in range(len(probs))}
        print(label_probs)
        score = signed_sent_score(label_probs)
        label = label_from_signed(score)
        method = "absa"
     
    else:
        # For the roberta sentiment pipeline, truncate using character count as approximation
        # or better yet, use the pipeline's tokenizer
        if len(text) > 500:  # Conservative character limit
            text = text[:500]
            
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
        print(label_probs)
        score = signed_sent_score(label_probs)
        label = label_from_signed(score)    
        method = "generic"

    try:
        if len(text) > 500:
            text = text[:500]
        check_sarcasm, sarcasm_score = analyze_sarcasm(text, sarcasm_detection_pipeline)
    except Exception as e:
        check_sarcasm, sarcasm_score = False, 0.0
    if check_sarcasm and abs(sarcasm_score) > 0.5  :
        score = -0.75 * score
        label = label_from_signed(score)
    return {"label": label, "score": score, "method": method}

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