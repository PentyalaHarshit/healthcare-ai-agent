from datetime import datetime, timedelta
import requests

from database import SessionLocal, Doctor
from google_calendar import create_appointment_event


def urgency_agent(bayesian_risk, ml_risk):
    """Determines urgency level based on risk scores."""
    bayesian_risk_pct = bayesian_risk.get("risk_percentage", 0)
    ml_risk_pct = ml_risk.get("risk_percentage", 0)
    
    # Average risk across Bayesian and ML
    avg_risk = (bayesian_risk_pct + ml_risk_pct) / 2

    if avg_risk >= 85:
        return {
            "urgency": "Emergency",
            "action": "Do not wait for appointment. Go to hospital immediately or call emergency services.",
            "schedule_type": "immediate"
        }

    if avg_risk >= 60:
        return {
            "urgency": "Urgent",
            "action": "Book earliest available doctor appointment today.",
            "schedule_type": "today"
        }

    if avg_risk >= 30:
        return {
            "urgency": "Soon",
            "action": "Schedule appointment within 2-3 days.",
            "schedule_type": "soon"
        }

    return {
        "urgency": "Normal",
        "action": "Schedule routine appointment within 1-2 weeks.",
        "schedule_type": "routine"
    }


def appointment_agent(xai_analysis):
    """Selects an appropriate doctor based on recommended specialty."""
    db = SessionLocal()
    
    specialty = xai_analysis.get("recommended_specialty", "General Physician")

    doctors = db.query(Doctor).filter(
        Doctor.specialty == specialty,
        Doctor.available == True
    ).all()

    db.close()

    if not doctors:
        return {
            "specialty": specialty,
            "doctor_found": False,
            "message": f"No available {specialty} found. Recommending General Physician."
        }

    # Select first available doctor
    selected_doctor = doctors[0]

    return {
        "specialty": specialty,
        "doctor_found": True,
        "doctor_id": selected_doctor.id,
        "doctor_name": selected_doctor.name,
        "experience": selected_doctor.experience,
        "available_time": selected_doctor.available_time
    }


def calendar_agent(patient, doctor, xai_analysis, urgency):
    """Creates a calendar event for the appointment."""
    if not doctor.get("doctor_found"):
        return {"calendar_event_created": False}

    # Determine appointment time based on urgency
    urgency_type = urgency.get("schedule_type", "routine")
    now = datetime.now()

    if urgency_type == "immediate":
        appointment_time = now + timedelta(hours=1)
    elif urgency_type == "today":
        appointment_time = now + timedelta(hours=2)
    elif urgency_type == "soon":
        appointment_time = now + timedelta(days=2)
    else:  # routine
        appointment_time = now + timedelta(days=7)

    try:
        event = create_appointment_event(
            patient_name=patient.name,
            doctor_name=doctor.get("doctor_name"),
            specialty=doctor.get("specialty", "General"),
            appointment_date=appointment_time.strftime("%Y-%m-%d"),
            appointment_time=appointment_time.strftime("%I:%M %p"),
            urgency=urgency.get("urgency", "Normal"),
            duration_minutes=30
        )
        return {
            "calendar_event_created": True,
            "event_id": event.get("id"),
            "appointment_time": appointment_time.isoformat(),
            "doctor": doctor.get("doctor_name"),
            "urgency": urgency_type
        }
    except Exception as e:
        return {
            "calendar_event_created": False,
            "error": str(e),
            "appointment_time": appointment_time.isoformat()
        }
