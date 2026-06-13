"""
Healthcare AI — Main Application
Upgrades wired in:
  1. Vector RAG        (rag_engine.py  — ChromaDB + sentence-transformers)
  2. Multi-Agent Crew  (crew_agents.py — 6-agent pipeline)
  3. Knowledge Graph   (knowledge_graph.py — NetworkX)
  4. Medical Image AI  (image_ai.py    — OpenCV/Pillow)
  5. Voice Agent       (voice_agent.py — Whisper STT + ElevenLabs TTS)
  6. Real ML Risk      (ml_risk_model.py — XGBoost/LightGBM)
  7. XAI Dashboard     (ml_risk_model.py get_shap_explanation)
"""
import uuid
import os
import logging
import json
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Request, Form, UploadFile, File, Query
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from auth import authenticate_user, create_access_token, require_roles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001")

app = FastAPI(title="Healthcare AI — Upgraded Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

sessions: dict = {}


@app.on_event("startup")
def startup_database():
    from database import init_db
    init_db()

HOSPITALS = [
    "Texas Health Frisco",
    "Baylor Scott & White Plano",
    "Medical City Plano",
    "Texas Health Plano",
    "Children's Medical Center Plano",
]

LOCATION_HOSPITALS = {
    "plano": [
        "Medical City Plano",
        "Baylor Scott & White Plano",
        "Texas Health Frisco",
    ],
    "frisco": [
        "Texas Health Frisco",
        "Baylor Scott & White Plano",
        "Medical City Plano",
    ],
}


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def infer_disease(symptoms: str, specialty: str | None = None) -> str:
    text = symptoms.lower()
    if "chest pain" in text or "shortness of breath" in text or "high blood pressure" in text or "hypertension" in text:
        return "Cardiovascular Risk"
    if "diabetes" in text or "blood sugar" in text:
        return "Diabetes"
    if "fever" in text or "cough" in text:
        return "Respiratory Infection"
    if "headache" in text or "dizziness" in text:
        return "Neurological Symptoms"
    if "stomach" in text or "nausea" in text or "vomiting" in text:
        return "Gastrointestinal Symptoms"
    return specialty or "General Symptoms"


def get_or_create_patient(name: str = "Patient", location: str | None = None, history: str | None = None):
    from database import Patient, SessionLocal

    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.name == name).first()
        if not patient:
            patient = Patient(name=name, location=location, medical_history=history, created_at=now_iso())
            db.add(patient)
            db.commit()
            db.refresh(patient)
        elif location or history:
            if location:
                patient.location = location
            if history:
                patient.medical_history = history
            db.commit()
            db.refresh(patient)
        patient_id = patient.id
    finally:
        db.close()
    return patient_id


def save_report_record(
    *,
    patient_name: str,
    hospital: str | None,
    symptoms: str,
    risk_score: int,
    priority: str,
    specialty: str,
    recommendation: str,
    details: dict | None = None,
) -> None:
    from database import Report, SessionLocal

    patient_id = get_or_create_patient(patient_name)
    db = SessionLocal()
    try:
        db.add(Report(
            patient_id=patient_id,
            patient_name=patient_name,
            hospital=hospital,
            symptoms=symptoms,
            disease=infer_disease(symptoms, specialty),
            specialty=specialty,
            risk_score=risk_score,
            priority=priority,
            recommendation=recommendation,
            details=json.dumps(details or {}, default=str),
            generated_at=now_iso(),
        ))
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning(f"Could not save report record: {exc}")
    finally:
        db.close()


def save_chat_history(session_id: str | None, message: str, response: dict) -> None:
    if not session_id:
        return

    from database import ChatHistory, SessionLocal

    reply = response.get("reply") or json.dumps({
        key: value for key, value in response.items()
        if key not in {"xai_details", "rag_medical_context", "kg_relationships"}
    }, default=str)

    db = SessionLocal()
    try:
        db.add(ChatHistory(
            patient_id=get_or_create_patient("Patient"),
            session_id=session_id,
            message=message,
            reply=reply,
            timestamp=now_iso(),
        ))
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning(f"Could not save chat history: {exc}")
    finally:
        db.close()


def finalize_chat_response(message: str, response: dict) -> dict:
    save_chat_history(response.get("session_id"), message, response)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Database Setup
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class BookRequest(BaseModel):
    slot_id: int


class SparkRiskPredictionRequest(BaseModel):
    age: int = 55
    chest_pain: bool = False
    diabetes: bool = False
    hypertension: bool = False
    shortness_of_breath: bool = False
    blood_pressure: int = 120


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helper functions (used by /analyze and /chat)
# ─────────────────────────────────────────────────────────────────────────────
def risk_prediction_model(symptoms: str, age: int = 55) -> int:
    s = symptoms.lower()
    risk = 0
    if "chest pain" in s:         risk += 40
    if "diabetes" in s:           risk += 25
    if "breathing" in s or "shortness of breath" in s: risk += 30
    if "high bp" in s or "blood pressure" in s: risk += 20
    if "hypertension" in s:       risk += 15
    if "fever" in s:              risk += 15
    if age >= 60:                 risk += 10
    return min(risk, 100)


def choose_specialty(symptoms: str) -> str:
    s = symptoms.lower()
    if "chest pain" in s or "diabetes" in s or "hypertension" in s:
        return "Cardiology"
    if "breathing" in s or "shortness of breath" in s:
        return "Pulmonology"
    if "headache" in s or "dizziness" in s:
        return "Neurology"
    return "General Physician"


def get_available_doctors(hospital: str, specialty: str) -> list:
    from database import SessionLocal, Doctor, Slot

    db = SessionLocal()
    try:
        rows = db.query(
            Slot.id, Doctor.name, Doctor.hospital, Doctor.specialty, Slot.date, Slot.time
        ).join(Doctor, Doctor.id == Slot.doctor_id).filter(
            Doctor.hospital.ilike(hospital),
            Doctor.specialty.ilike(specialty),
            Slot.available == 1,
        ).all()
    finally:
        db.close()

    return [
        {"slot_id": r[0], "doctor_name": r[1], "hospital": r[2],
         "specialty": r[3], "date": r[4], "time": r[5]}
        for r in rows
    ]


def get_hospitals_with_specialty(specialty: str) -> list:
    from database import SessionLocal, Doctor, Slot

    db = SessionLocal()
    try:
        rows = db.query(Doctor.hospital).join(
            Slot, Doctor.id == Slot.doctor_id
        ).filter(
            Doctor.specialty.ilike(specialty),
            Slot.available == 1,
        ).distinct().all()
    finally:
        db.close()

    return [r[0] for r in rows]


def get_hospital_reply() -> str:
    hospital_lines = "<br>".join(f"{idx}. {name}" for idx, name in enumerate(HOSPITALS, start=1))
    return f"<b>Available Hospitals</b><br><br>{hospital_lines}<br><br>Please select a hospital by typing its name."


def get_location_from_message(message: str) -> str | None:
    text = message.lower()
    for location in LOCATION_HOSPITALS:
        if location in text:
            return location
    return None


def is_hospital_list_request(message: str) -> bool:
    text = message.lower()
    list_words = ("list", "available", "show", "give", "nearby", "recommend")
    return "hospital" in text and any(word in text for word in list_words)


def is_hospital_name(message: str) -> bool:
    text = message.strip().lower()
    return any(text == hospital.lower() for hospital in HOSPITALS)


def resolve_hospital_name(message: str) -> str | None:
    text = message.strip().lower()
    for hospital in HOSPITALS:
        if text == hospital.lower():
            return hospital
    return None


def has_symptom_signal(message: str) -> bool:
    text = message.lower()
    symptom_words = (
        "i have", "i am having", "symptom", "pain", "fever", "diabetes",
        "chest", "breathing", "shortness of breath", "headache", "dizziness",
        "cough", "blood pressure", "hypertension", "rash", "stomach",
    )
    return any(word in text for word in symptom_words)


def risk_priority(risk_score: int) -> tuple[str, str]:
    if risk_score >= 90:
        return "Emergency", "Seek emergency care immediately."
    if risk_score >= 60:
        return "High", "Schedule an urgent appointment."
    return "Normal", "Schedule a routine consultation."


def recommend_hospitals(specialty: str, location: str | None = "plano") -> list:
    available = get_hospitals_with_specialty(specialty)
    if not available:
        available = HOSPITALS

    preferred = LOCATION_HOSPITALS.get((location or "").lower(), HOSPITALS)
    ranked = [hospital for hospital in preferred if hospital in available]
    ranked.extend(hospital for hospital in available if hospital not in ranked)
    ranked.extend(hospital for hospital in HOSPITALS if hospital not in ranked)
    return ranked[:3]


def format_recommended_hospitals(hospitals: list[str]) -> str:
    return "<br>".join(f"{idx}. {name}" for idx, name in enumerate(hospitals, start=1))


def format_doctor_recommendation(hospital: str, specialty: str, doctors: list) -> str:
    specialty_labels = {
        "Cardiology": "Cardiologists",
        "Pulmonology": "Pulmonologists",
        "Neurology": "Neurologists",
        "Endocrinology": "Endocrinologists",
        "Pediatrics": "Pediatricians",
        "General": "General Physicians",
        "General Physician": "General Physicians",
    }
    specialty_label = specialty_labels.get(specialty, f"{specialty} doctors")
    if not doctors:
        return (
            f"<b>{hospital}</b><br><br>"
            f"No available {specialty_label} were found at this hospital right now."
        )

    doctor_lines = "<br>".join(f"- {doctor['doctor_name']}" for doctor in doctors)
    slot_lines = "<br>".join(f"{doctor['time']}" for doctor in doctors)
    return (
        f"<b>{hospital}</b><br><br>"
        f"<b>Available {specialty_label} ({len(doctors)}):</b><br>{doctor_lines}<br><br>"
        f"<b>Available Slots:</b><br>{slot_lines}"
    )


def get_chat_xai_details(symptoms: str, age: int = 55) -> list:
    details = []
    s = symptoms.lower()
    if "chest pain" in s:
        details.append({"name": "Chest Pain", "points": 40})
    if "shortness of breath" in s or "breathing" in s:
        details.append({"name": "Shortness of Breath", "points": 30})
    if "high bp" in s or "blood pressure" in s:
        details.append({"name": "High Blood Pressure", "points": 20})
    if "hypertension" in s:
        details.append({"name": "Hypertension", "points": 15})
    if "diabetes" in s:
        details.append({"name": "Diabetes", "points": 25})
    if "fever" in s:
        details.append({"name": "Fever", "points": 15})
    if age >= 60:
        details.append({"name": f"Age ({age})", "points": 10})
    return details


def analyze_symptoms_for_chat(symptoms: str) -> dict:
    risk_score = risk_prediction_model(symptoms)
    specialty = choose_specialty(symptoms)
    priority, recommendation = risk_priority(risk_score)
    return {
        "symptoms": symptoms,
        "risk_score": risk_score,
        "priority": priority,
        "recommended_specialty": specialty,
        "recommendation": recommendation,
        "xai_explanation": [],
        "xai_details": get_chat_xai_details(symptoms),
        "rag_medical_context": _rag_fallback(symptoms),
        "kg_relationships": [],
        "disclaimer": "This is not a medical diagnosis. For emergencies, call 911.",
    }


def rag_agent(query: str) -> list:
    """Upgraded: uses semantic RAG engine."""
    try:
        from rag_engine import retrieve_guidelines
        docs = retrieve_guidelines(query, top_k=3)
        return [d["text"][:200] for d in docs] if docs else _rag_fallback(query)
    except Exception:
        return _rag_fallback(query)


def _rag_fallback(query: str) -> list:
    MEDICAL_KNOWLEDGE = [
        "Chest pain with diabetes may indicate elevated cardiovascular risk.",
        "Breathing difficulty with chest pain may require urgent evaluation.",
        "High blood pressure can increase the risk of heart disease.",
        "Fever and cough may indicate infection or respiratory illness.",
        "Patients with severe chest pain should seek immediate medical care.",
    ]
    q_words = set(query.lower().split())
    scored = [(doc, len(q_words & set(doc.lower().split()))) for doc in MEDICAL_KNOWLEDGE]
    scored.sort(key=lambda x: -x[1])
    return [d for d, s in scored[:3] if s > 0]


def get_xai_details(symptoms: str, age: int = 55) -> list:
    """Use ML model SHAP explanation if available, else rule-based."""
    try:
        from ml_risk_model import get_shap_explanation
        return get_shap_explanation(symptoms.split(", "), [], age)
    except Exception:
        details = []
        s = symptoms.lower()
        if "chest pain" in s:           details.append({"name": "Chest Pain", "points": 40})
        if "shortness of breath" in s:  details.append({"name": "Shortness of Breath", "points": 30})
        if "high bp" in s or "blood pressure" in s: details.append({"name": "High Blood Pressure", "points": 20})
        if "diabetes" in s:             details.append({"name": "Diabetes", "points": 25})
        if "fever" in s:                details.append({"name": "Fever", "points": 15})
        if age >= 60:                   details.append({"name": f"Age ({age})", "points": 10})
        return details


# ─────────────────────────────────────────────────────────────────────────────
# Auth Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}


@app.get("/login-page",  response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/patient-page", response_class=HTMLResponse)
def patient_page(request: Request):
    return templates.TemplateResponse(request=request, name="patient.html")

@app.get("/doctor-page",  response_class=HTMLResponse)
def doctor_page(request: Request):
    return templates.TemplateResponse(request=request, name="doctor.html")

@app.get("/admin-page",   response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse(request=request, name="admin.html")

@app.get("/report-page",  response_class=HTMLResponse)
def report_page(request: Request):
    return templates.TemplateResponse(request=request, name="report.html")

@app.get("/doctor-dashboard")
def doctor_dashboard(current_user=Depends(require_roles(["doctor", "admin"]))):
    return {"message": "Doctor dashboard access granted", "user": current_user}

@app.get("/patient-dashboard")
def patient_dashboard(current_user=Depends(require_roles(["patient", "admin"]))):
    return {"message": "Patient dashboard access granted", "user": current_user}

@app.get("/admin-dashboard")
def admin_dashboard(current_user=Depends(require_roles(["admin"]))):
    return {"message": "Admin dashboard access granted", "user": current_user}


@app.get("/hospital-analytics", response_class=HTMLResponse)
def hospital_analytics_page(request: Request):
    return templates.TemplateResponse(request=request, name="hospital_analytics.html")


@app.get("/api/hospital-analytics")
def api_hospital_analytics():
    from hospital_analytics import get_hospital_analytics
    return get_hospital_analytics()


@app.post("/api/hospital-analytics/export-hdfs")
def api_export_hospital_analytics_to_hdfs():
    from hospital_analytics import export_postgres_to_hdfs
    return export_postgres_to_hdfs()


@app.get("/api/hospital-analytics/spark")
def api_hospital_analytics_spark():
    from hospital_analytics import run_spark_hdfs_analytics
    return run_spark_hdfs_analytics()


@app.get("/analytics/common-diseases")
def analytics_common_diseases():
    from hospital_analytics import common_diseases_from_hdfs
    return common_diseases_from_hdfs()


@app.get("/analytics/busy-doctors")
def analytics_busy_doctors():
    from hospital_analytics import busy_doctors_from_hdfs
    return busy_doctors_from_hdfs()


@app.get("/analytics/hospital-load")
def analytics_hospital_load():
    from hospital_analytics import hospital_load_from_hdfs
    return hospital_load_from_hdfs()


@app.get("/analytics/emergency-trends")
def analytics_emergency_trends():
    from hospital_analytics import emergency_trends_from_hdfs
    return emergency_trends_from_hdfs()


@app.post("/ml/risk/train")
def ml_train_risk_model():
    from healthcare_spark_ml import train_risk_model_from_hdfs
    return train_risk_model_from_hdfs()


@app.post("/ml/risk/predict")
def ml_predict_risk(data: SparkRiskPredictionRequest):
    from healthcare_spark_ml import predict_risk_with_spark_ml
    return predict_risk_with_spark_ml(data.dict())


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade #5 — Voice Agent endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/patient/voice/input")
async def voice_input(audio: UploadFile = File(...)):
    """
    Receive audio recording from browser (webm/ogg/wav),
    transcribe via Whisper, return transcript.
    """
    from voice_agent import transcribe_audio

    try:
        audio_bytes = await audio.read()
        content_type = audio.content_type or "audio/webm"
        result = transcribe_audio(audio_bytes, content_type)

        if result.get("method") == "unavailable":
            raise HTTPException(
                status_code=503,
                detail="Speech recognition unavailable. Install openai-whisper.",
            )

        return {
            "transcript": result.get("transcript", ""),
            "language": result.get("language", "en"),
            "method": result.get("method", "whisper"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice input error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/patient/voice/tts")
async def text_to_speech(text: str = Query(..., max_length=500)):
    """Convert text to speech audio. Returns audio/mpeg bytes."""
    from voice_agent import synthesize_speech

    audio_bytes = synthesize_speech(text)
    if audio_bytes is None:
        raise HTTPException(status_code=503, detail="TTS service unavailable")
    return Response(content=audio_bytes, media_type="audio/mpeg")


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade #4 — Medical Image AI endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/patient/images/upload")
async def upload_medical_image(
    image: UploadFile = File(...),
    modality: str = Query("xray", description="xray | mri | ct | skin"),
):
    """
    Accept a medical image, run AI analysis, return findings.
    """
    from image_ai import analyse_medical_image

    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if image.content_type and image.content_type.lower() not in allowed:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images accepted")

    try:
        image_bytes = await image.read()
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 20 MB)")

        result = analyse_medical_image(image_bytes, modality)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade #3 — Knowledge Graph endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/knowledge-graph")
def knowledge_graph_data():
    """Return graph nodes/edges for visualisation."""
    from knowledge_graph import get_graph_json
    return get_graph_json()


@app.get("/api/knowledge-graph/query")
def knowledge_graph_query(symptoms: str = "", history: str = ""):
    """Query the knowledge graph for a given symptom string."""
    from knowledge_graph import query_knowledge_graph
    return query_knowledge_graph(symptoms, history)


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade #1+2+6+7 — Enhanced /analyze (Crew + Vector RAG + XGBoost + SHAP)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(request: Request, symptoms: str = Form(...)):
    """
    Full upgraded pipeline:
    symptoms → 6-agent crew → XAI report → Multi-LLM verification
    """
    from crew_agents import run_healthcare_crew
    from ml_risk_model import predict_ml_risk, get_shap_explanation

    hospital = patient_data.get("hospital", "Texas Health Frisco")

    # Run full crew
    try:
        crew_report = run_healthcare_crew(
            symptoms_text=symptoms,
            history_text="",
            patient_name="Patient",
        )
    except Exception as e:
        logger.error(f"Crew pipeline error: {e}")
        crew_report = {}

    # Risk score
    risk = crew_report.get("risk_score") or risk_prediction_model(symptoms)
    risk = int(min(risk, 100))

    if risk >= 90:   priority = "Emergency"
    elif risk >= 60: priority = "High"
    else:            priority = "Normal"

    # XAI / SHAP details
    xai_details = get_shap_explanation(symptoms.split(", "), [])
    if not xai_details:
        xai_details = get_xai_details(symptoms)

    # Doctors
    specialty = crew_report.get("recommended_specialty") or choose_specialty(symptoms)
    doctors = crew_report.get("available_doctors") or get_available_doctors(hospital, specialty)

    if doctors:
        doctor = f"{doctors[0]['doctor_name']} ({doctors[0]['specialty']}) at {doctors[0]['time']} ({doctors[0]['date']})"
        slot_id = doctors[0]["slot_id"]
    else:
        fallback = get_available_doctors(hospital, "General Physician")
        if fallback:
            doctor = f"{fallback[0]['doctor_name']} (General Physician - Fallback) at {fallback[0]['time']} ({fallback[0]['date']})"
            slot_id = fallback[0]["slot_id"]
        else:
            doctor = "No doctors available at this time."
            slot_id = None

    # RAG context
    rag_context = rag_agent(symptoms)

    # Knowledge graph relationships
    kg_relationships = crew_report.get("kg_relationships", [])

    # ML probabilities for dashboard
    try:
        ml_result = predict_ml_risk(symptoms.split(", "), [])
        ml_probs = ml_result.get("probabilities", {})
    except Exception:
        ml_probs = {}

    save_report_record(
        patient_name="Patient",
        hospital=hospital,
        symptoms=symptoms,
        risk_score=risk,
        priority=priority,
        specialty=specialty,
        recommendation=crew_report.get("recommendation", ""),
        details={
            "xai_details": xai_details,
            "rag_context": rag_context,
            "kg_relationships": kg_relationships,
            "ml_probabilities": ml_probs,
        },
    )

    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={
            "hospital": hospital,
            "symptoms": symptoms,
            "risk": risk,
            "priority": priority,
            "doctor": doctor,
            "slot_id": slot_id,
            "xai_details": xai_details,
            "rag_medical_context": rag_context,
            "kg_relationships": kg_relationships,
            "ml_probabilities": ml_probs,
            "crew_report": crew_report,
            "recommendation": crew_report.get("recommendation", ""),
            "urgency": crew_report.get("urgency", "Normal"),
        },
    )


@app.get("/api/multi-llm-verify")
def api_multi_llm_verify(symptoms: str = Query(...), history: str = "", age: int = 55):
    """
    Direct API endpoint to run parallel Multi-LLM Triage Consensus Verification.
    """
    try:
        from multi_llm_verifier import run_multi_llm_verification
        return run_multi_llm_verification(symptoms, history, age)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ─────────────────────────────────────────────────────────────────────────────
# Booking
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/book")
def book_appointment(data: BookRequest):
    from database import SessionLocal, Doctor, Slot

    db = SessionLocal()
    try:
        row = db.query(
            Slot, Doctor.id, Doctor.name, Doctor.hospital, Doctor.specialty, Slot.date, Slot.time
        ).join(Doctor, Doctor.id == Slot.doctor_id).filter(
            Slot.id == data.slot_id,
            Slot.available == 1,
        ).first()

        if not row:
            return {"status": "failed", "message": "Slot not available or does not exist"}

        slot, doctor_id, doctor_name, hospital, specialty, appointment_date, appointment_time = row
        slot.available = 0
        from database import Appointment
        db.add(Appointment(
            patient_id=get_or_create_patient("Patient"),
            doctor_id=doctor_id,
            patient_name="Patient",
            doctor_name=doctor_name,
            hospital=hospital,
            specialty=specialty,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            symptoms=None,
            urgency="Scheduled",
            status="Confirmed",
            created_at=now_iso(),
        ))
        db.commit()
    finally:
        db.close()

    # Create calendar event
    try:
        from google_calendar import create_appointment_event
        calendar_event = create_appointment_event(
            patient_name="Patient",
            doctor_name=doctor_name,
            specialty=specialty,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            urgency="Scheduled",
        )
    except Exception as e:
        logger.warning(f"Google Calendar event creation failed: {e}")
        calendar_event = None

    return {
        "status": "success",
        "message": "Appointment booked successfully",
        "calendar_event": calendar_event
    }



# ─────────────────────────────────────────────────────────────────────────────
# Healthcare multi-turn chat agent
# ─────────────────────────────────────────────────────────────────────────────
def router_agent(message: str) -> str:
    health_words = ["medical", "health", "doctor", "hospital", "pain", "fever",
                    "diabetes", "symptom", "chest", "breathing", "headache"]
    return "healthcare" if any(w in message.lower() for w in health_words) else "general"


def healthcare_agent(session_id: str, message: str) -> dict:
    session = sessions[session_id]

    if is_hospital_list_request(message):
        location = get_location_from_message(message)
        if location:
            nearby = LOCATION_HOSPITALS[location]
            hospital_lines = "<br>".join(f"- {hospital}" for hospital in nearby)
            return {"session_id": session_id, "agent": "Healthcare Agent",
                    "reply": f"<b>Location: {location.title()}</b><br><br><b>Nearby Hospitals:</b><br>{hospital_lines}<br><br>Please select a hospital by typing its name."}
        return {"session_id": session_id, "agent": "Healthcare Agent",
                "reply": get_hospital_reply()}

    if session["step"] == "ask_symptoms":
        analysis = analyze_symptoms_for_chat(message)
        location = get_location_from_message(message) or session.get("location") or "plano"
        hospitals = recommend_hospitals(analysis["recommended_specialty"], location)
        session.update({
            "step": "ask_hospital",
            "symptoms": message,
            "analysis": analysis,
            "recommended_hospitals": hospitals,
            "location": location,
        })

        save_report_record(
            patient_name="Patient",
            hospital=None,
            symptoms=message,
            risk_score=analysis["risk_score"],
            priority=analysis["priority"],
            specialty=analysis["recommended_specialty"],
            recommendation=analysis["recommendation"],
            details={"source": "chat"},
        )

        return {
            "session_id": session_id,
            "agent": "Healthcare Agent",
            "reply": (
                f"<b>Risk:</b> {analysis['priority']}<br>"
                f"<b>Specialty:</b> {analysis['recommended_specialty']}<br><br>"
                f"<b>Recommended Hospitals:</b><br>"
                f"{format_recommended_hospitals(hospitals)}<br><br>"
                "Which hospital would you prefer?"
            ),
            **analysis,
            "recommended_hospitals": hospitals,
        }

    if session["step"] == "ask_hospital":
        hospital = resolve_hospital_name(message)
        if not hospital:
            return {"session_id": session_id, "agent": "Healthcare Agent",
                    "reply": f"I could not match that hospital. Please choose one of these:<br><br>{format_recommended_hospitals(session.get('recommended_hospitals') or HOSPITALS)}"}

        analysis = session.get("analysis") or analyze_symptoms_for_chat(session.get("symptoms", ""))
        specialty = analysis["recommended_specialty"]
        doctors = get_available_doctors(hospital, specialty)
        session.update({"step": "ask_appointment", "hospital": hospital, "available_doctors": doctors})

        reply = format_doctor_recommendation(hospital, specialty, doctors)
        if doctors:
            reply += "<br><br>Please choose a slot to book your appointment."
        else:
            alternatives = [h for h in recommend_hospitals(specialty, session.get("location")) if h.lower() != hospital.lower()]
            session["recommended_hospitals"] = alternatives
            reply += f"<br><br>Other recommended hospitals:<br>{format_recommended_hospitals(alternatives)}"

        return {
            "session_id": session_id,
            "agent": "Healthcare Agent",
            "reply": reply,
            "hospital": hospital,
            **analysis,
            "available_doctors": doctors,
            "suggested_hospitals": [h for h in recommend_hospitals(specialty, session.get("location")) if h.lower() != hospital.lower()],
        }

    if session["step"] == "ask_appointment":
        hospital = session.get("hospital")
        analysis = session.get("analysis") or {}
        doctors = session.get("available_doctors") or []
        return {
            "session_id": session_id,
            "agent": "Healthcare Agent",
            "reply": format_doctor_recommendation(hospital, analysis.get("recommended_specialty", "General"), doctors),
            "hospital": hospital,
            **analysis,
            "available_doctors": doctors,
        }

    return {"session_id": session_id,
            "reply": "Session completed. Start a new session for another request."}


@app.post("/chat")
def chat(data: ChatRequest):
    message = data.message

    if not data.session_id:
        session_id = str(uuid.uuid4())
        route = router_agent(message)
        if route == "healthcare":
            location = get_location_from_message(message)
            sessions[session_id] = {"route": "healthcare", "step": "ask_symptoms",
                                     "hospital": None, "symptoms": None, "location": location}
            if is_hospital_list_request(message):
                return finalize_chat_response(message, healthcare_agent(session_id, message) | {"router_agent": "Healthcare Agent selected"})
            if has_symptom_signal(message):
                return finalize_chat_response(message, healthcare_agent(session_id, message) | {"router_agent": "Healthcare Agent selected"})
            return finalize_chat_response(message, {"session_id": session_id, "router_agent": "Healthcare Agent selected",
                    "reply": "Please describe your symptoms so I can analyze your risk and recommend the right hospital."})
        sessions[session_id] = {"route": "general", "step": "general_chat"}
        return finalize_chat_response(message, {"session_id": session_id, "router_agent": "General Agent selected",
                "reply": "General answer: Please ask your question."})

    session_id = data.session_id
    if session_id not in sessions:
        return {"error": "Invalid session_id"}

    if sessions[session_id]["route"] == "general":
        if router_agent(message) == "healthcare":
            sessions[session_id] = {"route": "healthcare", "step": "ask_symptoms",
                                     "hospital": None, "symptoms": None,
                                     "location": get_location_from_message(message)}
            if is_hospital_list_request(message) or has_symptom_signal(message):
                return finalize_chat_response(message, healthcare_agent(session_id, message) | {"router_agent": "Healthcare Agent selected"})
            return finalize_chat_response(message, {"session_id": session_id, "router_agent": "Healthcare Agent selected",
                    "reply": "Please describe your symptoms so I can analyze your risk and recommend the right hospital."})
        return finalize_chat_response(message, {"session_id": session_id,
                "reply": "I'm here to help with medical bookings. Describe symptoms to begin."})

    if sessions[session_id]["route"] == "healthcare":
        return finalize_chat_response(message, healthcare_agent(session_id, message))

    return finalize_chat_response(message, {"session_id": session_id, "reply": "General agent response"})


# ─────────────────────────────────────────────────────────────────────────────
# HTML page routes
# ─────────────────────────────────────────────────────────────────────────────
patient_data: dict = {}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/hospital")
async def hospital(request: Request, hospital: str = Form(...)):
    patient_data["hospital"] = hospital
    return templates.TemplateResponse(
        request=request, name="symptoms.html", context={"hospital": hospital})


@app.get("/hospital")
async def get_hospital(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")


@app.get("/analyze")
async def get_analyze(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")
