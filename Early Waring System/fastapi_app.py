from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model_combined = joblib.load('xgb_model_best.pkl')

class StudentData(BaseModel):
    Attendance: float
    Grades: float
    Homework_Streak: int
    Feedback_Behavior: int
    Weekly_Test_Scores: float
    Attention_Test_Scores: float
    Ragging: int
    Finance_Issue: int
    Mental_Health_Issue: int
    Physical_Health_Issue: int
    Discrimination_Based_on_Gender: int
    Physical_Disability: int

@app.post("/predict")
def predict_risk(data: StudentData):

    data_dict = data.dict()
    df = pd.DataFrame([data_dict])
    
    prediction = model_combined.predict(df)[0]
    
    if prediction == 0:
        risk_status_str = "Low Risk"
        risk_type = "No Risk"
    else:
        risk_status_str = "High Risk"
        risk_type = ["Academic Risk", "Financial Risk", "Mental Health Risk", "Bullying Risk"][prediction]
    
    return {
        "Risk_Status": risk_status_str,
        "Risk_Type": risk_type
    }
