from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from model_train import tokenize_sentence

app = FastAPI()

with open("comment_classificat_model.pkl", "rb") as model_file:
    model = joblib.load(model_file)

class InputData(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: InputData):
    preprocessed_text = tokenize_sentence(data.text)
    prediction = model.predict([" ".join(preprocessed_text)])
    return {"prediction": prediction[0]}