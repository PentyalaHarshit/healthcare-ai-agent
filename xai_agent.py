from database import SessionLocal, HealthRule


def xai_patient_analysis_agent(patient):
    """XAI agent for patient symptom and history analysis."""
    db = SessionLocal()
    
    symptoms_text = " ".join(patient.symptoms).lower()
    history_text = " ".join(patient.medical_history).lower()

    rules = db.query(HealthRule).all()

    matched_rules = []
    explanation = []
    risk_score = 0
    specialty_scores = {}

    priority_rank = {
        "Normal": 1,
        "Medium": 2,
        "High": 3
    }

    final_priority = "Normal"

    for rule in rules:
        keyword = rule.keyword.lower()

        if rule.category == "symptom":
            source_text = symptoms_text
        else:
            source_text = history_text

        if keyword in source_text:
            matched_rules.append({
                "keyword": rule.keyword,
                "category": rule.category,
                "specialty": rule.specialty,
                "priority": rule.priority,
                "risk_points": rule.risk_points,
                "multiplier": rule.multiplier,
                "explanation": rule.explanation
            })
            explanation.append(rule.explanation)
            risk_score += rule.risk_points

            specialty = rule.specialty
            specialty_scores[specialty] = (
                specialty_scores.get(specialty, 0)
                + rule.risk_points
            )

            if priority_rank[rule.priority] > priority_rank[final_priority]:
                final_priority = rule.priority

    db.close()

    if not matched_rules:
        return {
            "recommended_specialty": "General Physician",
            "priority": "Normal",
            "risk_score": 10,
            "confidence": 0.10,
            "matched_rules": [],
            "xai_explanation": [
                "No matching health rule found. General physician consultation recommended."
            ]
        }

    recommended_specialty = max(
        specialty_scores,
        key=specialty_scores.get
    )

    confidence = min(risk_score / 100, 0.95)

    return {
        "recommended_specialty": recommended_specialty,
        "priority": final_priority,
        "risk_score": risk_score,
        "confidence": confidence,
        "matched_rules": matched_rules,
        "xai_explanation": explanation
    }
