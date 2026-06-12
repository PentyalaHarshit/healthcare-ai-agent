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
import sqlite3
import uuid
import os
import logging
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

DB = str(BASE_DIR / "hospital.db")
sessions: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Database Setup
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, hospital TEXT, specialty TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS slots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER, date TEXT, time TEXT, available INTEGER
        )""")
    cur.execute("DELETE FROM doctors")
    cur.execute("DELETE FROM slots")
    for tbl in ("doctors", "slots"):
        try:
            cur.execute(f"DELETE FROM sqlite_sequence WHERE name='{tbl}'")
        except sqlite3.OperationalError:
            pass

    cur.executemany(
        "INSERT INTO doctors(id, name, hospital, specialty) VALUES (?, ?, ?, ?)",
        [
            (1, "Dr. Smith",  "Texas Health Frisco", "Cardiology"),
            (2, "Dr. Kumar",  "Texas Health Frisco", "Cardiology"),
            (3, "Dr. Lee",    "Texas Health Frisco", "Pulmonology"),
            (4, "Dr. Brown",  "Texas Health Frisco", "General"),
            (5, "Dr. Patel",  "Baylor Hospital",     "Cardiology"),
            (6, "Dr. Nguyen", "Texas Health Frisco", "Neurology"),
            (7, "Dr. Okafor", "Baylor Hospital",     "Endocrinology"),
        ],
    )
    cur.executemany(
        "INSERT INTO slots(doctor_id, date, time, available) VALUES (?, ?, ?, ?)",
        [
            (1, "2026-06-20", "10:00 AM", 1),
            (1, "2026-06-20", "04:30 PM", 1),
            (2, "2026-06-21", "09:00 AM", 1),
            (3, "2026-06-20", "11:30 AM", 1),
            (4, "2026-06-22", "02:00 PM", 1),
            (5, "2026-06-20", "03:00 PM", 1),
            (6, "2026-06-21", "01:00 PM", 1),
            (7, "2026-06-22", "10:30 AM", 1),
        ],
    )
    conn.commit()
    conn.close()


init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class BookRequest(BaseModel):
    slot_id: int


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
    return "General"


def get_available_doctors(hospital: str, specialty: str) -> list:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT slots.id, doctors.name, doctors.hospital, doctors.specialty,
               slots.date, slots.time
        FROM doctors
        JOIN slots ON doctors.id = slots.doctor_id
        WHERE LOWER(doctors.hospital) = LOWER(?)
          AND LOWER(doctors.specialty) = LOWER(?)
          AND slots.available = 1
    """, (hospital, specialty))
    rows = cur.fetchall()
    conn.close()
    return [
        {"slot_id": r[0], "doctor_name": r[1], "hospital": r[2],
         "specialty": r[3], "date": r[4], "time": r[5]}
        for r in rows
    ]


def get_hospitals_with_specialty(specialty: str) -> list:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT doctors.hospital FROM doctors
        JOIN slots ON doctors.id = slots.doctor_id
        WHERE LOWER(doctors.specialty) = LOWER(?) AND slots.available = 1
    """, (specialty,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


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
    symptoms → 6-agent crew → XAI report
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
        fallback = get_available_doctors(hospital, "General")
        if fallback:
            doctor = f"{fallback[0]['doctor_name']} (General - Fallback) at {fallback[0]['time']} ({fallback[0]['date']})"
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


# ─────────────────────────────────────────────────────────────────────────────
# Booking
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/book")
def book_appointment(data: BookRequest):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT slots.id, doctors.name, doctors.hospital, doctors.specialty,
               slots.date, slots.time
        FROM doctors
        JOIN slots ON doctors.id = slots.doctor_id
        WHERE slots.id = ? AND slots.available = 1
    """, (data.slot_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"status": "failed", "message": "Slot not available or does not exist"}

    cur.execute("UPDATE slots SET available = 0 WHERE id = ? AND available = 1", (data.slot_id,))
    conn.commit()
    updated = cur.rowcount
    conn.close()

    if updated == 0:
        return {"status": "failed", "message": "Slot not available"}

    # Create calendar event
    try:
        from google_calendar import create_appointment_event
        calendar_event = create_appointment_event(
            patient_name="Patient",
            doctor_name=row[1],
            specialty=row[3],
            appointment_date=row[4],
            appointment_time=row[5],
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

    if session["step"] == "ask_hospital":
        session["hospital"] = message
        session["step"] = "ask_symptoms"
        return {"session_id": session_id, "agent": "Healthcare Agent",
                "reply": "Please describe your symptoms."}

    if session["step"] == "ask_symptoms":
        session["symptoms"] = message
        session["step"] = "completed"
        hospital = session["hospital"]
        symptoms = message

        try:
            from crew_agents import run_healthcare_crew
            crew = run_healthcare_crew(symptoms, "", "Patient")
            risk_score = int(crew.get("risk_score", risk_prediction_model(symptoms)))
        except Exception:
            risk_score = risk_prediction_model(symptoms)
            crew = {}

        if risk_score >= 90:   priority, recommendation = "Emergency", "Seek emergency care immediately."
        elif risk_score >= 60: priority, recommendation = "High", "Schedule an urgent appointment."
        else:                  priority, recommendation = "Normal", "Schedule a routine consultation."

        specialty = crew.get("recommended_specialty") or choose_specialty(symptoms)
        doctors = get_available_doctors(hospital, specialty)
        xai_details = get_xai_details(symptoms)
        rag_context = rag_agent(symptoms)

        res = {
            "session_id": session_id,
            "agent": "Healthcare Agent",
            "hospital": hospital,
            "symptoms": symptoms,
            "risk_score": risk_score,
            "priority": priority,
            "recommended_specialty": specialty,
            "xai_explanation": crew.get("xai_analysis", {}).get("xai_explanation", []),
            "xai_details": xai_details,
            "rag_medical_context": rag_context,
            "available_doctors": doctors,
            "recommendation": recommendation,
            "kg_relationships": crew.get("kg_relationships", []),
            "disclaimer": "This is not a medical diagnosis. For emergencies, call 911.",
        }

        if not doctors:
            alt = [h for h in get_hospitals_with_specialty(specialty) if h.lower() != hospital.lower()]
            res["suggested_hospitals"] = alt

        return res

    return {"session_id": session_id,
            "reply": "Session completed. Start a new session for another request."}


@app.post("/chat")
def chat(data: ChatRequest):
    message = data.message

    if not data.session_id:
        session_id = str(uuid.uuid4())
        route = router_agent(message)
        if route == "healthcare":
            sessions[session_id] = {"route": "healthcare", "step": "ask_hospital",
                                     "hospital": None, "symptoms": None}
            return {"session_id": session_id, "router_agent": "Healthcare Agent selected",
                    "reply": "What hospital would you like to visit?"}
        sessions[session_id] = {"route": "general", "step": "general_chat"}
        return {"session_id": session_id, "router_agent": "General Agent selected",
                "reply": "General answer: Please ask your question."}

    session_id = data.session_id
    if session_id not in sessions:
        return {"error": "Invalid session_id"}

    if sessions[session_id]["route"] == "general":
        if router_agent(message) == "healthcare":
            sessions[session_id] = {"route": "healthcare", "step": "ask_hospital",
                                     "hospital": None, "symptoms": None}
            return {"session_id": session_id, "router_agent": "Healthcare Agent selected",
                    "reply": "What hospital would you like to visit?"}
        return {"session_id": session_id,
                "reply": "I'm here to help with medical bookings. Describe symptoms to begin."}

    if sessions[session_id]["route"] == "healthcare":
        return healthcare_agent(session_id, message)

    return {"reply": "General agent response"}


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

