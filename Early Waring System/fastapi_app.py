
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load both models
model_risk_status = joblib.load('xgb_model_risk_status.pkl')
model_risk_type = joblib.load('xgb_model_risk_type.pkl')

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
    
    # Predict risk status
    risk_status = model_risk_status.predict(df)[0]
    risk_status_str = "High Risk" if risk_status == 1 else "Low Risk"
    
    # Predict type of risk only if the student is at risk
    risk_type = "N/A"
    if risk_status == 1:
        risk_type = model_risk_type.predict(df)[0]
    
    return {
        "Risk_Status": risk_status_str,
        "Risk_Type": risk_type
    }
