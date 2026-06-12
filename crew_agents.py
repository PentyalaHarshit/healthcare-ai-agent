"""
Multi-Agent Healthcare Crew — Upgrade #2
6-agent pipeline:
  1. Symptom Agent    — parse & classify symptoms
  2. Risk Agent       — compute ML + Bayesian risk
  3. RAG Agent        — retrieve medical guidelines
  4. Knowledge Agent  — query knowledge graph
  5. Doctor Agent     — select best available doctor
  6. Report Agent     — assemble final report
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Agent 1: Symptom Agent ────────────────────────────────────────────────────
def symptom_agent(symptoms_text: str, history_text: str) -> dict:
    """Parse raw text into classified symptom list."""
    CRITICAL = ["chest pain", "shortness of breath", "arm pain", "palpitations", "severe dizziness"]
    HIGH = ["breathing", "hypertension", "high blood pressure", "blurred vision", "sweating"]
    MEDIUM = ["headache", "dizziness", "diabetes", "frequent urination", "fever", "nausea", "vomiting", "stomach pain"]

    text = (symptoms_text + " " + history_text).lower()
    critical_found = [k for k in CRITICAL if k in text]
    high_found = [k for k in HIGH if k in text]
    medium_found = [k for k in MEDIUM if k in text]

    if critical_found:
        severity = "Critical"
    elif high_found:
        severity = "High"
    elif medium_found:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "agent": "Symptom Agent",
        "symptoms_text": symptoms_text,
        "history_text": history_text,
        "critical_symptoms": critical_found,
        "high_symptoms": high_found,
        "medium_symptoms": medium_found,
        "severity": severity,
    }


# ── Agent 2: Risk Agent ───────────────────────────────────────────────────────
def risk_agent(symptom_result: dict,
               age: int = 55, bp: int = 120,
               heart_rate: int = 75, cholesterol: int = 180) -> dict:
    """Compute ML risk and Bayesian risk in parallel."""
    symptoms = symptom_result["symptoms_text"].split()
    history = symptom_result["history_text"].split()

    # ML risk
    try:
        from ml_risk_model import predict_ml_risk
        ml = predict_ml_risk(symptoms, history, age, bp, heart_rate, cholesterol)
    except Exception as e:
        logger.warning(f"ML risk failed: {e}")
        text = symptom_result["symptoms_text"].lower()
        score = sum([
            40 if "chest pain" in text else 0,
            30 if "shortness of breath" in text else 0,
            25 if "diabetes" in text else 0,
            20 if "hypertension" in text or "high blood pressure" in text else 0,
        ])
        ml = {"ml_risk_prediction": "High" if score >= 60 else "Medium" if score >= 30 else "Low",
              "confidence": 0.7, "risk_percentage": min(score, 99), "probabilities": {}}

    # Bayesian risk
    from bayesian_agent import bayesian_risk_agent
    from xai_agent import xai_patient_analysis_agent

    class _FakePatient:
        def __init__(self, s, h):
            self.symptoms = s.split(", ") if s else ["general"]
            self.medical_history = h.split(", ") if h else []

    patient = _FakePatient(symptom_result["symptoms_text"], symptom_result["history_text"])
    xai = xai_patient_analysis_agent(patient)
    bayesian = bayesian_risk_agent(xai.get("matched_rules", []))

    combined_risk = max(ml.get("risk_percentage", 0), bayesian.get("risk_percentage", 0))

    return {
        "agent": "Risk Agent",
        "ml_risk": ml,
        "bayesian_risk": bayesian,
        "xai_analysis": xai,
        "combined_risk_percentage": round(combined_risk, 1),
        "risk_level": "High" if combined_risk >= 60 else "Medium" if combined_risk >= 30 else "Low",
        "recommended_specialty": xai.get("recommended_specialty", "General Physician"),
    }


# ── Agent 3: RAG Agent ────────────────────────────────────────────────────────
def rag_crew_agent(symptoms_text: str, history_text: str) -> dict:
    """Retrieve semantic medical guidelines."""
    from rag_engine import rag_medical_explanation
    result = rag_medical_explanation(
        symptoms_text.split(", "),
        history_text.split(", "),
    )
    return {
        "agent": "RAG Agent",
        **result,
    }


# ── Agent 4: Knowledge Graph Agent ───────────────────────────────────────────
def knowledge_agent(symptoms_text: str, history_text: str) -> dict:
    """Query the knowledge graph for symptom→specialty relationships."""
    from knowledge_graph import query_knowledge_graph
    result = query_knowledge_graph(symptoms_text, history_text)
    return {
        "agent": "Knowledge Agent",
        **result,
    }


# ── Agent 5: Doctor Agent ─────────────────────────────────────────────────────
def doctor_agent(specialty: str) -> dict:
    """Select best available doctor for the given specialty."""
    import sqlite3
    from pathlib import Path

    db_path = str(Path(__file__).resolve().parent / "hospital.db")
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT slots.id, doctors.name, doctors.hospital, doctors.specialty,
                   slots.date, slots.time
            FROM doctors
            JOIN slots ON doctors.id = slots.doctor_id
            WHERE LOWER(doctors.specialty) = LOWER(?) AND slots.available = 1
            LIMIT 3
        """, (specialty,))
        rows = cur.fetchall()
        conn.close()

        doctors = [
            {"slot_id": r[0], "doctor_name": r[1], "hospital": r[2],
             "specialty": r[3], "date": r[4], "time": r[5]}
            for r in rows
        ]
        return {
            "agent": "Doctor Agent",
            "specialty": specialty,
            "available_doctors": doctors,
            "doctor_found": len(doctors) > 0,
        }
    except Exception as e:
        return {"agent": "Doctor Agent", "specialty": specialty,
                "available_doctors": [], "doctor_found": False, "error": str(e)}


# ── Agent 6: Report Agent ─────────────────────────────────────────────────────
def report_agent(symptom_result, risk_result, rag_result, kg_result, doctor_result,
                 patient_name: str = "Patient") -> dict:
    """Assemble final structured report from all agent outputs."""
    risk_pct = risk_result["combined_risk_percentage"]

    if risk_pct >= 85:
        urgency = "Emergency"
        recommendation = "⚠️ Do NOT wait. Go to emergency services immediately."
    elif risk_pct >= 60:
        urgency = "Urgent"
        recommendation = "Book the earliest available appointment today."
    elif risk_pct >= 30:
        urgency = "Soon"
        recommendation = "Schedule within 2-3 days."
    else:
        urgency = "Routine"
        recommendation = "Routine consultation within 1-2 weeks."

    return {
        "agent": "Report Agent",
        "patient": patient_name,
        "generated_at": datetime.now().isoformat(),
        "severity": symptom_result["severity"],
        "urgency": urgency,
        "recommendation": recommendation,
        "risk_score": risk_pct,
        "risk_level": risk_result["risk_level"],
        "recommended_specialty": risk_result["recommended_specialty"],
        "ml_risk": risk_result["ml_risk"],
        "bayesian_risk": risk_result["bayesian_risk"],
        "xai_analysis": risk_result["xai_analysis"],
        "rag_context": rag_result.get("retrieved_guidelines", []),
        "rag_explanation": rag_result.get("rag_explanation", ""),
        "kg_relationships": kg_result.get("relationships", []),
        "kg_specialty": kg_result.get("recommended_specialty", "General Physician"),
        "available_doctors": doctor_result.get("available_doctors", []),
        "critical_symptoms": symptom_result["critical_symptoms"],
        "disclaimer": "This is not a medical diagnosis. For emergencies, call 911 immediately.",
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────
def run_healthcare_crew(
    symptoms_text: str,
    history_text: str = "",
    patient_name: str = "Patient",
    age: int = 55,
    bp: int = 120,
    heart_rate: int = 75,
    cholesterol: int = 180,
) -> dict:
    """
    Full 6-agent pipeline.
    Returns final_report dict.
    """
    logger.info(f"🏥 Healthcare Crew starting for: {patient_name}")

    try:
        s = symptom_agent(symptoms_text, history_text)
        logger.info(f"[1/6] Symptom Agent → severity={s['severity']}")
    except Exception as e:
        logger.error(f"Symptom agent failed: {e}")
        s = {"agent": "Symptom Agent", "symptoms_text": symptoms_text,
             "history_text": history_text, "critical_symptoms": [],
             "high_symptoms": [], "medium_symptoms": [], "severity": "Unknown"}

    try:
        r = risk_agent(s, age, bp, heart_rate, cholesterol)
        logger.info(f"[2/6] Risk Agent → risk={r['combined_risk_percentage']}%")
    except Exception as e:
        logger.error(f"Risk agent failed: {e}")
        r = {"agent": "Risk Agent", "combined_risk_percentage": 50, "risk_level": "Medium",
             "recommended_specialty": "General Physician", "ml_risk": {}, "bayesian_risk": {},
             "xai_analysis": {"matched_rules": [], "xai_explanation": []}}

    try:
        rag = rag_crew_agent(symptoms_text, history_text)
        logger.info(f"[3/6] RAG Agent → {len(rag.get('retrieved_guidelines', []))} docs")
    except Exception as e:
        logger.error(f"RAG agent failed: {e}")
        rag = {"agent": "RAG Agent", "retrieved_guidelines": [], "rag_explanation": ""}

    try:
        kg = knowledge_agent(symptoms_text, history_text)
        logger.info(f"[4/6] Knowledge Agent → specialty={kg.get('recommended_specialty')}")
    except Exception as e:
        logger.error(f"Knowledge agent failed: {e}")
        kg = {"agent": "Knowledge Agent", "recommended_specialty": "General Physician",
              "relationships": [], "kg_risk_score": 0}

    specialty = r.get("recommended_specialty", kg.get("recommended_specialty", "General Physician"))

    try:
        doc = doctor_agent(specialty)
        logger.info(f"[5/6] Doctor Agent → {len(doc.get('available_doctors', []))} doctors found")
    except Exception as e:
        logger.error(f"Doctor agent failed: {e}")
        doc = {"agent": "Doctor Agent", "specialty": specialty,
               "available_doctors": [], "doctor_found": False}

    try:
        report = report_agent(s, r, rag, kg, doc, patient_name)
        logger.info(f"[6/6] Report Agent → urgency={report['urgency']}")
    except Exception as e:
        logger.error(f"Report agent failed: {e}")
        report = {"error": str(e), "recommendation": "Please consult a doctor."}

    return report
