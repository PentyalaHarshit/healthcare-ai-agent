from main import app
from pydantic import BaseModel
import requests

from langchain_core.tools import tool
from google_calendar import create_google_calendar_event

MCP_SERVER_URL = "http://127.0.0.1:8001"


class PatientRequest(BaseModel):
    name: str
    age: int
    symptoms: list[str]
    medical_history: list[str]


@tool
def find_doctors_from_mcp(specialty: str):
    """Find available doctors by specialty from MCP doctor DB server."""
    response = requests.get(
        f"{MCP_SERVER_URL}/tools/find_doctors",
        params={"specialty": specialty}
    )
    return response.json()["result"]


def patient_analysis_agent(patient: PatientRequest):
    symptoms = " ".join(patient.symptoms).lower()
    history = " ".join(patient.medical_history).lower()

    if "chest pain" in symptoms or "shortness of breath" in symptoms:
        return {
            "specialty": "Cardiology",
            "priority": "High",
            "reason": "Chest pain or breathing issue detected."
        }

    if "diabetes" in history:
        return {
            "specialty": "Endocrinology",
            "priority": "Medium",
            "reason": "Diabetes history detected."
        }

    if "headache" in symptoms:
        return {
            "specialty": "Neurology",
            "priority": "Medium",
            "reason": "Headache detected."
        }

    return {
        "specialty": "General Physician",
        "priority": "Normal",
        "reason": "General consultation recommended."
    }


def appointment_agent(analysis):
    doctors = find_doctors_from_mcp.invoke(analysis["specialty"])

    if not doctors:
        return None

    return sorted(doctors, key=lambda d: d["experience"], reverse=True)[0]


def report_agent(patient, analysis, doctor, calendar_event):
    return {
        "patient": patient.name,
        "age": patient.age,
        "analysis": analysis,
        "selected_doctor": doctor,
        "calendar_event": calendar_event,
        "note": "This is not medical diagnosis. Please consult a licensed doctor."
    }


@app.post("/book-appointment")
def book_appointment(patient: PatientRequest):
    analysis = patient_analysis_agent(patient)

    doctor = appointment_agent(analysis)

    if doctor is None:
        return {
            "message": "No doctor available",
            "recommended_specialty": analysis["specialty"]
        }

    calendar_event = create_google_calendar_event(
        patient_name=patient.name,
        doctor_name=doctor["name"],
        specialty=doctor["specialty"],
        time=doctor["available_time"]
    )

    return report_agent(patient, analysis, doctor, calendar_event)