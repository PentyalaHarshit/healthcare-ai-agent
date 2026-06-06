import sqlite3
import uuid
import os
from fastapi import FastAPI, Depends, HTTPException, Request, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from auth import authenticate_user, create_access_token, require_roles

# MCP Server URL — overridden by Docker env var MCP_SERVER_URL
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001")


app = FastAPI(title="Healthcare Multi-Turn Agent")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the base directory of the application
BASE_DIR = Path(__file__).resolve().parent

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
# Configure templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DB = str(BASE_DIR / "hospital.db")
sessions = {}


# -----------------------------
# Database Setup
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        hospital TEXT,
        specialty TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS slots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER,
        date TEXT,
        time TEXT,
        available INTEGER
    )
    """)

    cur.execute("DELETE FROM doctors")
    cur.execute("DELETE FROM slots")
    try:
        cur.execute("DELETE FROM sqlite_sequence WHERE name='doctors'")
        cur.execute("DELETE FROM sqlite_sequence WHERE name='slots'")
    except sqlite3.OperationalError:
        pass

    doctors = [
        (1, "Dr. Smith", "Texas Health Frisco", "Cardiology"),
        (2, "Dr. Kumar", "Texas Health Frisco", "Cardiology"),
        (3, "Dr. Lee", "Texas Health Frisco", "Pulmonology"),
        (4, "Dr. Brown", "Texas Health Frisco", "General"),
        (5, "Dr. Patel", "Baylor Hospital", "Cardiology"),
    ]

    cur.executemany(
        "INSERT INTO doctors(id, name, hospital, specialty) VALUES (?, ?, ?, ?)",
        doctors
    )

    slots = [
        (1, "2026-06-06", "10:00 AM", 1),
        (1, "2026-06-06", "04:30 PM", 1),
        (2, "2026-06-07", "09:00 AM", 1),
        (3, "2026-06-06", "11:30 AM", 1),
        (4, "2026-06-08", "02:00 PM", 1),
        (5, "2026-06-06", "03:00 PM", 1),
    ]

    cur.executemany(
        "INSERT INTO slots(doctor_id, date, time, available) VALUES (?, ?, ?, ?)",
        slots
    )

    conn.commit()
    conn.close()


init_db()


# -----------------------------
# Request Models
# -----------------------------
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class BookRequest(BaseModel):
    slot_id: int


# -----------------------------
# Router Agent
# -----------------------------
def router_agent(message: str):
    msg = message.lower()

    health_words = ["medical", "health", "doctor", "hospital", "pain", "fever", "diabetes"]

    if any(word in msg for word in health_words):
        return "healthcare"

    return "general"


# -----------------------------
# Risk Prediction Model
# -----------------------------
def risk_prediction_model(symptoms: str, age: int = 55):
    symptoms = symptoms.lower()
    risk = 0

    if "chest pain" in symptoms:
        risk += 40

    if "diabetes" in symptoms:
        risk += 25

    if "breathing" in symptoms or "shortness of breath" in symptoms:
        risk += 30

    if "high bp" in symptoms or "blood pressure" in symptoms:
        risk += 20

    if "fever" in symptoms:
        risk += 15

    if age >= 60:
        risk += 10

    return min(risk, 100)


# -----------------------------
# XAI Agent
# -----------------------------
def xai_agent(symptoms: str, age: int = 55):
    symptoms = symptoms.lower()
    reasons = []

    if "chest pain" in symptoms:
        reasons.append("Chest pain increased risk by 40")

    if "diabetes" in symptoms:
        reasons.append("Diabetes increased cardiovascular risk by 25")

    if "breathing" in symptoms or "shortness of breath" in symptoms:
        reasons.append("Breathing difficulty increased risk by 30")

    if "high bp" in symptoms or "blood pressure" in symptoms:
        reasons.append("High blood pressure increased risk by 20")

    if "fever" in symptoms:
        reasons.append("Fever increased risk by 15")

    if age >= 60:
        reasons.append("Age above 60 increased risk by 10")

    return reasons


def get_xai_details(symptoms: str, age: int = 55):
    symptoms = symptoms.lower()
    details = []

    if "chest pain" in symptoms:
        details.append({"name": "Chest Pain", "points": 40})

    if "breathing" in symptoms or "shortness of breath" in symptoms:
        details.append({"name": "Shortness of Breath", "points": 30})

    if "high bp" in symptoms or "blood pressure" in symptoms:
        details.append({"name": "High Blood Pressure", "points": 20})

    if "diabetes" in symptoms:
        details.append({"name": "Diabetes", "points": 25})

    if "fever" in symptoms:
        details.append({"name": "Fever", "points": 15})

    if age >= 60:
        details.append({"name": "Age above 60", "points": 10})

    return details



# -----------------------------
# RAG Agent
# -----------------------------
MEDICAL_KNOWLEDGE = [
    "Chest pain with diabetes may indicate elevated cardiovascular risk.",
    "Breathing difficulty with chest pain may require urgent evaluation.",
    "High blood pressure can increase the risk of heart disease.",
    "Fever and cough may indicate infection or respiratory illness.",
    "Patients with severe chest pain should seek immediate medical care.",
    "Chest pain with breathing difficulty requires urgent evaluation.",
    "High blood pressure increases cardiovascular risk.",
    "Immediate cardiology consultation recommended."
]


def rag_agent(query: str):
    clean_query = "".join(c for c in query.lower() if c.isalnum() or c.isspace())
    query_words = set(clean_query.split())
    results = []

    for doc in MEDICAL_KNOWLEDGE:
        clean_doc = "".join(c for c in doc.lower() if c.isalnum() or c.isspace())
        doc_words = set(clean_doc.split())
        score = len(query_words.intersection(doc_words))

        if score > 0:
            results.append((doc, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in results[:3]]


# -----------------------------
# Doctor Selection Agent
# -----------------------------
def choose_specialty(symptoms: str):
    symptoms = symptoms.lower()

    if "chest pain" in symptoms or "diabetes" in symptoms:
        return "Cardiology"

    if "breathing" in symptoms or "shortness of breath" in symptoms:
        return "Pulmonology"

    return "General"


def get_available_doctors(hospital: str, specialty: str):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    SELECT
        slots.id,
        doctors.name,
        doctors.hospital,
        doctors.specialty,
        slots.date,
        slots.time
    FROM doctors
    JOIN slots ON doctors.id = slots.doctor_id
    WHERE LOWER(doctors.hospital) = LOWER(?)
      AND LOWER(doctors.specialty) = LOWER(?)
      AND slots.available = 1
    """, (hospital, specialty))

    rows = cur.fetchall()
    conn.close()

    doctors = []

    for row in rows:
        doctors.append({
            "slot_id": row[0],
            "doctor_name": row[1],
            "hospital": row[2],
            "specialty": row[3],
            "date": row[4],
            "time": row[5]
        })

    return doctors


def get_hospitals_with_specialty(specialty: str):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
    SELECT DISTINCT doctors.hospital
    FROM doctors
    JOIN slots ON doctors.id = slots.doctor_id
    WHERE LOWER(doctors.specialty) = LOWER(?)
      AND slots.available = 1
    """, (specialty,))
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows]


# -----------------------------
# Authentication & User Endpoints
# -----------------------------
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(
        form_data.username,
        form_data.password
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    token = create_access_token({
        "sub": user["username"],
        "role": user["role"]
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"]
    }


@app.get("/login-page", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html"
    )


@app.get("/patient-page", response_class=HTMLResponse)
def patient_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="patient.html"
    )


@app.get("/doctor-page", response_class=HTMLResponse)
def doctor_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="doctor.html"
    )


@app.get("/admin-page", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="admin.html"
    )


@app.get("/report-page", response_class=HTMLResponse)
def report_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="report.html"
    )


@app.get("/doctor-dashboard")
def doctor_dashboard(
    current_user=Depends(
        require_roles(["doctor", "admin"])
    )
):
    return {
        "message": "Doctor dashboard access granted",
        "user": current_user
    }


@app.get("/patient-dashboard")
def patient_dashboard(
    current_user=Depends(
        require_roles(["patient", "admin"])
    )
):
    return {
        "message": "Patient dashboard access granted",
        "user": current_user
    }


@app.get("/admin-dashboard")
def admin_dashboard(
    current_user=Depends(
        require_roles(["admin"])
    )
):
    return {
        "message": "Admin dashboard access granted",
        "user": current_user
    }


# -----------------------------
# Appointment Booking Endpoint
# -----------------------------
@app.post("/book")
def book_appointment(data: BookRequest):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
        "UPDATE slots SET available = 0 WHERE id = ? AND available = 1",
        (data.slot_id,)
    )

    conn.commit()
    updated = cur.rowcount
    conn.close()

    if updated == 0:
        return {
            "status": "failed",
            "message": "Slot not available"
        }

    return {
        "status": "success",
        "message": "Appointment booked successfully"
    }


# -----------------------------
# Healthcare Agent
# -----------------------------
def healthcare_agent(session_id: str, message: str):
    session = sessions[session_id]

    if session["step"] == "ask_hospital":
        session["hospital"] = message
        session["step"] = "ask_symptoms"

        return {
            "session_id": session_id,
            "agent": "Healthcare Agent",
            "reply": "Please describe your symptoms."
        }

    if session["step"] == "ask_symptoms":
        session["symptoms"] = message
        session["step"] = "completed"

        hospital = session["hospital"]
        symptoms = session["symptoms"]

        risk_score = risk_prediction_model(symptoms)
        xai = xai_agent(symptoms)
        rag_context = rag_agent(symptoms)

        specialty = choose_specialty(symptoms)
        doctors = get_available_doctors(hospital, specialty)

        if risk_score >= 90:
            priority = "Emergency"
            recommendation = "Seek emergency medical care immediately."
        elif risk_score >= 60:
            priority = "High"
            recommendation = "Schedule an urgent doctor appointment."
        else:
            priority = "Normal"
            recommendation = "Schedule a normal consultation."

        res = {
            "session_id": session_id,
            "agent": "Healthcare Agent",
            "hospital": hospital,
            "symptoms": symptoms,
            "risk_score": risk_score,
            "priority": priority,
            "recommended_specialty": specialty,
            "xai_explanation": xai,
            "xai_details": get_xai_details(symptoms),
            "rag_medical_context": rag_context,
            "available_doctors": doctors,
            "recommendation": recommendation,
            "disclaimer": "This is not a medical diagnosis. For emergencies, call emergency services immediately."
        }

        if not doctors:
            alt_hospitals = get_hospitals_with_specialty(specialty)
            alt_hospitals = [h for h in alt_hospitals if h.lower() != hospital.lower()]
            res["suggested_hospitals"] = alt_hospitals

        return res

    return {
        "session_id": session_id,
        "reply": "This session is completed. Start a new session for another request."
    }


# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
def chat(data: ChatRequest):
    message = data.message

    if not data.session_id:
        session_id = str(uuid.uuid4())
        route = router_agent(message)

        if route == "healthcare":
            sessions[session_id] = {
                "route": "healthcare",
                "step": "ask_hospital",
                "hospital": None,
                "symptoms": None
            }

            return {
                "session_id": session_id,
                "router_agent": "Healthcare Agent selected",
                "reply": "What hospital would you like to visit?"
            }

        sessions[session_id] = {
            "route": "general",
            "step": "general_chat"
        }
        return {
            "session_id": session_id,
            "router_agent": "General Agent selected",
            "reply": "General answer: Please ask your question."
        }

    session_id = data.session_id

    if session_id not in sessions:
        return {
            "error": "Invalid session_id"
        }

    if sessions[session_id]["route"] == "general":
        route = router_agent(message)
        if route == "healthcare":
            sessions[session_id] = {
                "route": "healthcare",
                "step": "ask_hospital",
                "hospital": None,
                "symptoms": None
            }
            return {
                "session_id": session_id,
                "router_agent": "Healthcare Agent selected",
                "reply": "What hospital would you like to visit?"
            }
        return {
            "session_id": session_id,
            "reply": "General agent response: I'm here to help with medical bookings. Type 'medical help' or ask about medical symptoms to begin."
        }

    if sessions[session_id]["route"] == "healthcare":
        return healthcare_agent(session_id, message)

    return {
        "reply": "General agent response"
    }


patient_data = {}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/hospital")
async def hospital(
    request: Request,
    hospital: str = Form(...)
):
    patient_data["hospital"] = hospital

    return templates.TemplateResponse(
        request=request,
        name="symptoms.html",
        context={
            "hospital": hospital
        }
    )


@app.post("/analyze")
async def analyze(
    request: Request,
    symptoms: str = Form(...)
):
    patient_data["symptoms"] = symptoms

    # Run the risk prediction using our risk model
    risk = risk_prediction_model(symptoms)

    if risk >= 90:
        priority = "Emergency"
    elif risk >= 60:
        priority = "High"
    else:
        priority = "Normal"

    specialty = choose_specialty(symptoms)
    selected_hospital = patient_data.get("hospital", "Texas Health Frisco")
    doctors = get_available_doctors(selected_hospital, specialty)
    
    if doctors:
        doctor = f"{doctors[0]['doctor_name']} ({doctors[0]['specialty']}) at {doctors[0]['time']} ({doctors[0]['date']})"
        slot_id = doctors[0]['slot_id']
    else:
        fallback_doctors = get_available_doctors(selected_hospital, "General")
        if fallback_doctors:
            doctor = f"{fallback_doctors[0]['doctor_name']} (General - Fallback) at {fallback_doctors[0]['time']} ({fallback_doctors[0]['date']})"
            slot_id = fallback_doctors[0]['slot_id']
        else:
            doctor = "No doctors available at this time."
            slot_id = None

    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={
            "hospital": selected_hospital,
            "symptoms": symptoms,
            "risk": risk,
            "priority": priority,
            "doctor": doctor,
            "slot_id": slot_id,
            "xai_details": get_xai_details(symptoms),
            "rag_medical_context": rag_agent(symptoms)
        }
    )