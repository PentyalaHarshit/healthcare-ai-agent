import os
import json
import logging
import requests
import concurrent.futures
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Ollama Endpoint Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_HOST}/api/generate"

# Configure models (names can be customized via env variables)
MODELS = {
    "phi": os.getenv("PHI_MODEL", "phi3"),
    "qwen": os.getenv("QWEN_MODEL", "qwen2.5"),
    "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-r1"),
}

# Clinical Triage Rules for Simulation Fallback
def get_clinical_baseline(symptoms_text: str) -> Dict[str, str]:
    """Helper to determine baseline urgency and specialty based on input symptoms."""
    s = symptoms_text.lower()
    
    # 1. Emergency triggers
    if any(x in s for x in ["chest pain", "arm pain", "palpitations", "pressure in chest", "severe chest"]):
        return {"urgency": "Emergency", "specialty": "Cardiology"}
    if any(x in s for x in ["shortness of breath", "breathing difficulty", "cannot breathe", "stridor"]):
        return {"urgency": "Emergency", "specialty": "Pulmonology"}
    
    # 2. High/Urgent triggers
    if any(x in s for x in ["dizziness", "blurred vision", "slurred speech", "numbness"]):
        return {"urgency": "Urgent", "specialty": "Neurology"}
    if any(x in s for x in ["high blood pressure", "hypertension", "bp"]):
        return {"urgency": "Urgent", "specialty": "Cardiology"}
    
    # 3. Soon triggers
    if any(x in s for x in ["diabetes", "frequent urination", "blood sugar"]):
        return {"urgency": "Soon", "specialty": "Endocrinology"}
    if any(x in s for x in ["stomach pain", "nausea", "vomiting", "diarrhea"]):
        return {"urgency": "Soon", "specialty": "Gastroenterology"}
    if any(x in s for x in ["fever", "cough", "sore throat", "fatigue"]):
        return {"urgency": "Soon", "specialty": "General Physician"}
        
    # 4. Routine fallback
    return {"urgency": "Routine", "specialty": "General Physician"}


def generate_simulated_response(model_key: str, symptoms: str, baseline: Dict[str, str]) -> Dict[str, Any]:
    """Generates a highly realistic simulated response reflecting the specific model's style and reasoning."""
    urgency = baseline["urgency"]
    specialty = baseline["specialty"]
    
    if model_key == "phi":
        explanation = (
            f"- Patient symptoms checked: {symptoms}\n"
            f"- Clinical evaluation indicates priority level: {urgency}.\n"
            f"- Recommendation: Immediate consultation with {specialty}.\n"
            f"- Core action: Check vital parameters, record ECG if cardiac symptoms are present.\n"
            f"- Follow-up: Rule out acute severe pathology within 24 hours."
        )
        return {
            "urgency": urgency,
            "specialty": specialty,
            "explanation": explanation,
            "confidence": 0.85,
            "raw_response": f"Urgency: {urgency}\nSpecialty: {specialty}\nExplanation:\n{explanation}"
        }
        
    elif model_key == "ollama":
        # Meta Llama persona
        explanation = (
            f"Based on the clinical description of your symptoms ({symptoms}), I highly recommend "
            f"consulting with a specialist in {specialty} as a priority. An urgency level of '{urgency}' "
            f"means we should not wait too long to get this professionally assessed. If you are experiencing "
            f"any secondary severe symptoms (such as radiation of pain or sudden weakness), please bypass standard "
            f"scheduling and proceed to urgent care immediately. Keep a close log of your symptoms in the meantime."
        )
        return {
            "urgency": urgency,
            "specialty": specialty,
            "explanation": explanation,
            "confidence": 0.80,
            "raw_response": f"Urgency: {urgency}\nSpecialty: {specialty}\nExplanation:\n{explanation}"
        }
        
    elif model_key == "qwen":
        explanation = (
            f"[Clinical Assessment] Analysis of symptoms ({symptoms}) indicates a triage category of {urgency}.\n"
            f"[Recommended Action] Refer directly to the Department of {specialty}.\n"
            f"[Safety Guidelines]\n"
            f"1. Monitor heart rate, blood pressure, and core temperature changes.\n"
            f"2. Avoid strenuous activity and seek rest immediately.\n"
            f"3. In case of sudden exacerbation of symptoms, visit the nearest emergency facility."
        )
        return {
            "urgency": urgency,
            "specialty": specialty,
            "explanation": explanation,
            "confidence": 0.90,
            "raw_response": f"Urgency: {urgency}\nSpecialty: {specialty}\nExplanation:\n{explanation}"
        }
        
    elif model_key == "deepseek":
        # DeepSeek persona with <think> tag
        think = (
            f"<think>\n"
            f"The patient is reporting symptoms: {symptoms}.\n"
            f"1. Evaluate clinical severity: Urgency determined as {urgency}.\n"
            f"2. Map to medical department: Primary system affected corresponds to {specialty}.\n"
            f"3. Formulate differential diagnostics: Must rule out life-threatening conditions (e.g. cardiovascular or severe respiratory event if applicable).\n"
            f"4. Decide recommendation: Suggest direct referral to {specialty} and outline precaution measures. Maintain strict safety guidelines.\n"
            f"</think>"
        )
        explanation = (
            f"Following a systematic clinical reasoning process, your symptoms ({symptoms}) are categorized "
            f"under the '{urgency}' priority bracket. It is advised to seek medical guidance from a {specialty} specialist. "
            f"Ensure you rest in a comfortable environment and have a caregiver nearby. If your breathing deteriorates "
            f"or pain increases, call emergency services immediately."
        )
        return {
            "urgency": urgency,
            "specialty": specialty,
            "explanation": explanation,
            "think": think,
            "confidence": 0.95,
            "raw_response": f"{think}\nUrgency: {urgency}\nSpecialty: {specialty}\nExplanation:\n{explanation}"
        }
        
    return {"urgency": "Soon", "specialty": "General Physician", "explanation": "Evaluation completed.", "confidence": 0.5}


def parse_model_response(raw_text: str, baseline: Dict[str, str]) -> Dict[str, Any]:
    """
    Parses LLM output text to extract Urgency, Specialty, and Explanation.
    Includes robust fallback parsing to scan for keywords.
    """
    text = raw_text.strip()
    
    # Check for DeepSeek thinking block
    think_block = ""
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>")
        think_block = parts[0].replace("<think>", "").strip()
        text = parts[1].strip() if len(parts) > 1 else ""
        
    urgency = None
    specialty = None
    explanation = text
    
    # 1. Parse line by line to look for prefixes
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith("urgency:"):
            val = line.split(":", 1)[1].strip()
            # Clean punctuation
            val = "".join(c for c in val if c.isalnum() or c.isspace()).strip()
            if val.title() in ["Emergency", "Urgent", "Soon", "Routine"]:
                urgency = val.title()
        elif line_lower.startswith("specialty:"):
            val = line.split(":", 1)[1].strip()
            val = "".join(c for c in val if c.isalnum() or c.isspace()).strip()
            if val:
                specialty = val.title()
        elif line_lower.startswith("explanation:") or line_lower.startswith("reasoning:"):
            continue
        else:
            cleaned_lines.append(line)
            
    if cleaned_lines:
        explanation = "\n".join(cleaned_lines).strip()
        
    # 2. Keyword fallback if not explicitly defined
    text_lower = text.lower()
    if not urgency:
        for opt in ["emergency", "urgent", "soon", "routine"]:
            if opt in text_lower:
                urgency = opt.title()
                break
                
    if not specialty:
        for opt in ["cardiology", "pulmonology", "neurology", "endocrinology", "gastroenterology", "general physician"]:
            if opt in text_lower:
                specialty = opt.title()
                break
                
    # 3. Hard fallback to baseline if still missing
    if not urgency:
        urgency = baseline["urgency"]
    if not specialty:
        specialty = baseline["specialty"]
    if not explanation or len(explanation) < 10:
        explanation = text if len(text) > 10 else f"Referral to {specialty} under urgency category {urgency}."
        
    res = {
        "urgency": urgency,
        "specialty": specialty,
        "explanation": explanation,
        "confidence": 0.80
    }
    if think_block:
        res["think"] = f"<think>\n{think_block}\n</think>"
        
    return res


def query_single_model(model_key: str, symptoms: str, history: str, age: int, baseline: Dict[str, str]) -> Dict[str, Any]:
    """Queries a single Ollama model, falling back to simulated output on failure."""
    model_name = MODELS.get(model_key, "llama3")
    
    prompt = (
        f"You are a clinical AI medical triage agent representing model '{model_key}'.\n"
        f"Patient details:\n"
        f"- Age: {age}\n"
        f"- Symptoms: {symptoms}\n"
        f"- Medical History: {history}\n\n"
        f"Analyze these details and provide your triage decision. You MUST respond in the following format:\n"
        f"Urgency: <Emergency | Urgent | Soon | Routine>\n"
        f"Specialty: <Cardiology | Pulmonology | Neurology | Gastroenterology | Endocrinology | General Physician>\n"
        f"Explanation: <Provide a brief explanation of your clinical reasoning>\n\n"
        f"Note: Be precise. If you are DeepSeek, write your chain of thought reasoning inside <think>...</think> first."
    )
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 300
        }
    }
    
    try:
        # Check connection with a fast timeout
        logger.info(f"Connecting to Ollama for {model_key} using model {model_name}...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=3.5)
        
        if response.status_code == 200:
            result_json = response.json()
            raw_response = result_json.get("response", "")
            parsed = parse_model_response(raw_response, baseline)
            parsed["mode"] = "live"
            parsed["raw_response"] = raw_response
            logger.info(f"✅ Live response received from Ollama model '{model_name}' for {model_key}")
            return parsed
        else:
            logger.warning(f"Ollama returned status {response.status_code} for model '{model_name}'")
            
    except requests.exceptions.RequestException as e:
        logger.debug(f"Ollama connection error or timeout for {model_key}: {e}")
        
    # Fallback to simulation
    simulated = generate_simulated_response(model_key, symptoms, baseline)
    simulated["mode"] = "simulated"
    logger.info(f"ℹ️ Simulated response generated for {model_key} (Ollama offline/model not pulled)")
    return simulated


def aggregate_majority_vote(predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes majority vote for urgency and specialty.
    Resolves ties using clinical safety priority.
    """
    urgency_ranks = {
        "Emergency": 4,
        "Urgent": 3,
        "Soon": 2,
        "Routine": 1
    }
    
    # Tally votes
    urgency_votes = {}
    specialty_votes = {}
    
    for model_key, pred in predictions.items():
        u = pred.get("urgency", "Soon")
        s = pred.get("specialty", "General Physician")
        
        urgency_votes[u] = urgency_votes.get(u, 0) + 1
        specialty_votes[s] = specialty_votes.get(s, 0) + 1
        
    # Determine consensus urgency
    max_u_votes = max(urgency_votes.values())
    winning_urgencies = [u for u, count in urgency_votes.items() if count == max_u_votes]
    
    if len(winning_urgencies) == 1:
        consensus_urgency = winning_urgencies[0]
    else:
        # Tie-breaker: clinical safety priority (highest severity rank)
        consensus_urgency = max(winning_urgencies, key=lambda u: urgency_ranks.get(u, 0))
        logger.info(f"Urgency tie resolved: {winning_urgencies} -> {consensus_urgency} (clinical safety rule)")
        
    # Determine consensus specialty
    max_s_votes = max(specialty_votes.values())
    winning_specialties = [s for s, count in specialty_votes.items() if count == max_s_votes]
    
    if len(winning_specialties) == 1:
        consensus_specialty = winning_specialties[0]
    else:
        # Tie-breaker: choose specialty associated with a model that voted for the consensus urgency
        possible_specialties = []
        for model_key, pred in predictions.items():
            if pred.get("urgency") == consensus_urgency:
                possible_specialties.append(pred.get("specialty"))
                
        # Find which of these had the most votes
        if possible_specialties:
            consensus_specialty = max(possible_specialties, key=lambda s: specialty_votes.get(s, 0))
        else:
            consensus_specialty = winning_specialties[0]
        logger.info(f"Specialty tie resolved: {winning_specialties} -> {consensus_specialty}")
        
    # Verified Medical Answer: Compile aggregate reasoning from consensus models
    consensus_models = []
    explanation_parts = []
    
    for model_key, pred in predictions.items():
        # A model is part of consensus if it agrees with either Urgency or Specialty
        if pred.get("urgency") == consensus_urgency and pred.get("specialty") == consensus_specialty:
            consensus_models.append(model_key.upper())
            explanation_parts.append(f"[{model_key.upper()}]: {pred.get('explanation')}")
            
    # If no model has both, pick models that agreed with Urgency
    if not consensus_models:
        for model_key, pred in predictions.items():
            if pred.get("urgency") == consensus_urgency:
                consensus_models.append(model_key.upper())
                explanation_parts.append(f"[{model_key.upper()}]: {pred.get('explanation')}")
                
    verified_explanation = (
        f"Consensus achieved by {', '.join(consensus_models)}.\n\n" + 
        "\n\n".join(explanation_parts)
    )
    
    return {
        "consensus_urgency": consensus_urgency,
        "consensus_specialty": consensus_specialty,
        "verified_explanation": verified_explanation,
        "urgency_votes": urgency_votes,
        "specialty_votes": specialty_votes,
        "consensus_models": consensus_models
    }


def run_multi_llm_verification(symptoms: str, history: str = "", age: int = 55) -> Dict[str, Any]:
    """Runs parallel LLM verification using ThreadPoolExecutor."""
    baseline = get_clinical_baseline(symptoms)
    
    predictions = {}
    live_count = 0
    
    # Run in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        future_to_model = {
            executor.submit(query_single_model, model_key, symptoms, history, age, baseline): model_key
            for model_key in MODELS.keys()
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model_key = future_to_model[future]
            try:
                data = future.result()
                predictions[model_key] = data
                if data.get("mode") == "live":
                    live_count += 1
            except Exception as exc:
                logger.error(f"{model_key} generated an exception: {exc}")
                # Ultimate fallback
                fallback = generate_simulated_response(model_key, symptoms, baseline)
                fallback["mode"] = "simulated"
                predictions[model_key] = fallback
                
    consensus = aggregate_majority_vote(predictions)
    
    return {
        "status": "success",
        "live_mode": live_count > 0,
        "live_models_count": live_count,
        "predictions": predictions,
        "consensus": consensus
    }
