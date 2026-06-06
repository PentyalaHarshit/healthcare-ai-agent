from fastapi import FastAPI
from database import init_db, SessionLocal, Doctor, HealthRule

app = FastAPI(title="Healthcare MCP Server")

init_db()


@app.get("/")
def home():
    return {"message": "Healthcare MCP Server running"}


@app.get("/tools/find_doctors")
def find_doctors(specialty: str):
    db = SessionLocal()

    doctors = db.query(Doctor).filter(
        Doctor.specialty == specialty,
        Doctor.available == True
    ).all()

    result = [
        {
            "id": d.id,
            "name": d.name,
            "specialty": d.specialty,
            "experience": d.experience,
            "available_time": d.available_time,
            "available": d.available
        }
        for d in doctors
    ]

    db.close()
    return {"tool": "find_doctors", "result": result}


@app.get("/tools/health_rules")
def health_rules():
    db = SessionLocal()

    rules = db.query(HealthRule).all()

    result = [
        {
            "keyword": r.keyword,
            "category": r.category,
            "specialty": r.specialty,
            "priority": r.priority,
            "risk_points": r.risk_points,
            "multiplier": r.multiplier,
            "explanation": r.explanation,
        }
        for r in rules
    ]

    db.close()
    return {"tool": "health_rules", "result": result}