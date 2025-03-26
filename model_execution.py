from fastapi import FastAPI, Query
import torch
import torch.nn.functional as F
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc  # Garbage Collection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Lazy Load Model
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_weights = torch.tensor([38.0, 1.0, 95.0], dtype=torch.float).to(device)

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model_name = "NikolaML/NewsAnalizer"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()

@app.get("/classify")
def classify_text(text: str = Query(..., description="Text to classify")):
    load_model()  # Load model only when needed

    # Tokenize input text (reduced max_length for efficiency)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)

    # Apply class weights
    weighted_probs = probs * class_weights
    weighted_probs /= weighted_probs.sum(dim=1, keepdim=True)

    # Prediction
    predicted_label = torch.argmax(weighted_probs, dim=1).item()
    confidence = weighted_probs[0, predicted_label].item()

    # Clean up memory
    del inputs, logits, probs, weighted_probs
    torch.cuda.empty_cache()
    gc.collect()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    return {"prediction": label_map[predicted_label], "confidence": round(confidence, 2)}

