from fastapi import FastAPI
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel

# ✅ Load Fine-Tuned Model
model_name = "NikolaML/NewsAnalizer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ Class Weights
class_weights = torch.tensor([38.0000, 1.0000, 95.0000], dtype=torch.float).to(device)

# ✅ FastAPI Setup
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/classify")
def classify_text(input_data: InputText):
    text = input_data.text

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits
    logits = outputs.logits

    # Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=1)

    # Apply class weights
    weighted_probs = probs * class_weights
    weighted_probs /= weighted_probs.sum(dim=1, keepdim=True)

    # Get prediction
    predicted_label = torch.argmax(weighted_probs, dim=1).item()
    confidence = weighted_probs[0, predicted_label].item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    return {"prediction": label_map[predicted_label], "confidence": round(confidence, 2)}
