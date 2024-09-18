
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model_combined = joblib.load('xgboost_multi_output_pipeline_model.pkl')

# Define the schema for input data
class StudentData(BaseModel):
    Attendance: float
    Grades: float
    Homework_Streak: int  # Adjusted schema
    Feedback_Behavior: int
    Weekly_Test_Scores: float  # Adjusted schema
    Attention_Test_Scores: float  # Adjusted schema
    Ragging: int
    Finance_Issue: int
    Mental_Health_Issue: int
    Physical_Health_Issue: int
    Discrimination_Based_on_Gender: int
    Physical_Disability: int
    Not_Interested: int
    Working_and_Studying: int
    School_Is_Far: int

@app.post("/predict")
def predict_risk(data: StudentData):
    try:
        # Convert input data into a DataFrame
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])

        # Rename columns to match model's expected input format
        df.rename(columns={
            'Homework_Streak': 'Homework Streak',
            'Weekly_Test_Scores': 'Weekly Test Scores',
            'Attention_Test_Scores': 'Attention Test Scores'
        }, inplace=True)

        # Check if all required columns are present in the DataFrame
        missing_columns = [col for col in [
            'Homework Streak', 'Weekly Test Scores', 'Attention Test Scores',
            'Feedback Behavior', 'Finance Issue', 'Mental Health Issue', 
            'Physical Health Issue', 'Discrimination Based on Gender', 
            'Physical Disability', 'Not Interested', 'Work and Learn', 
            'School is Far Off'
        ] if col not in df.columns]
        if missing_columns:
            return {"error": f"Missing columns: {missing_columns}"}

        # Predict risk status and type using the model
        prediction = model_combined.predict(df)[0]

        # Interpret the prediction result
        if prediction == 0:
            risk_status_str = "Low Risk"
            risk_type = "No Risk"
        else:
            risk_status_str = "High Risk"
            # Assuming the model outputs a class index for risk types
            risk_types = ["Academic Risk", "Financial Risk", "Mental Health Risk", "Bullying Risk"]
            risk_type = risk_types[prediction - 1] if prediction - 1 < len(risk_types) else "Unknown Risk Type"

        return {
            "Risk_Status": risk_status_str,
            "Risk_Type": risk_type
        }

    except KeyError as e:
        return {"error": f"KeyError during prediction: {e}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
