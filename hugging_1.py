from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = FastAPI()


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

class TextData(BaseModel):
    text: str

@app.post("/classify/")
async def classify_text(data: TextData):
    inputs = tokenizer(data.text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        model.config.id2label[predicted_class_id]
    return {"result": predicted_class_id}


# 이거 야놀자 그건데 컴 사양이 더 좋아야할듯.. 
# from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
# tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")

# prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {prompt}\nAssistant:\n"


# class TextData(BaseModel):
#     text: str

# @app.post("/yanolja/")
# async def classify_text(data: TextData):
#     inputs = tokenizer(prompt_template.format(prompt = data.text), return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=256)
#     output_text = tokenizer.batch._decode(outputs, skip_special_tokens=True)[0]
#     return {"result": output_text}