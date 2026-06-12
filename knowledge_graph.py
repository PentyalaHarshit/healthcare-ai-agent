"""
Knowledge Graph — Upgrade #3
Stores symptom/condition → specialty relationships using NetworkX.
Falls back to a hardcoded dict if networkx not installed.
"""
import logging

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _G = nx.DiGraph()

    # ── Nodes: symptoms / conditions / specialties ──────────────────────────
    symptoms = [
        "chest pain", "shortness of breath", "palpitations", "arm pain",
        "dizziness", "headache", "nausea", "blurred vision",
        "diabetes", "hypertension", "high blood pressure", "obesity",
        "fever", "cough", "fatigue", "frequent urination",
        "skin rash", "itching", "stomach pain", "vomiting", "joint pain",
    ]
    specialties = [
        "Cardiology", "Neurology", "Endocrinology", "Pulmonology",
        "Dermatology", "Gastroenterology", "General Physician",
    ]

    for s in symptoms:
        _G.add_node(s, type="symptom")
    for sp in specialties:
        _G.add_node(sp, type="specialty")

    # ── Edges: symptom → specialty (with risk weight) ───────────────────────
    edges = [
        ("chest pain",          "Cardiology",        {"weight": 40, "risk": "High"}),
        ("shortness of breath", "Cardiology",        {"weight": 30, "risk": "High"}),
        ("shortness of breath", "Pulmonology",       {"weight": 25, "risk": "High"}),
        ("palpitations",        "Cardiology",        {"weight": 25, "risk": "Medium"}),
        ("arm pain",            "Cardiology",        {"weight": 20, "risk": "Medium"}),
        ("hypertension",        "Cardiology",        {"weight": 15, "risk": "Medium"}),
        ("high blood pressure", "Cardiology",        {"weight": 15, "risk": "Medium"}),
        ("headache",            "Neurology",         {"weight": 20, "risk": "Medium"}),
        ("dizziness",           "Neurology",         {"weight": 15, "risk": "Medium"}),
        ("blurred vision",      "Neurology",         {"weight": 15, "risk": "Medium"}),
        ("diabetes",            "Endocrinology",     {"weight": 20, "risk": "Medium"}),
        ("frequent urination",  "Endocrinology",     {"weight": 15, "risk": "Medium"}),
        ("obesity",             "Endocrinology",     {"weight": 10, "risk": "Low"}),
        ("fever",               "General Physician", {"weight": 15, "risk": "Medium"}),
        ("cough",               "Pulmonology",       {"weight": 20, "risk": "Medium"}),
        ("fatigue",             "General Physician", {"weight": 10, "risk": "Low"}),
        ("skin rash",           "Dermatology",       {"weight": 10, "risk": "Low"}),
        ("itching",             "Dermatology",       {"weight": 8,  "risk": "Low"}),
        ("stomach pain",        "Gastroenterology",  {"weight": 20, "risk": "Medium"}),
        ("nausea",              "Gastroenterology",  {"weight": 12, "risk": "Low"}),
        ("vomiting",            "Gastroenterology",  {"weight": 15, "risk": "Medium"}),
        ("joint pain",          "General Physician", {"weight": 10, "risk": "Low"}),
    ]
    _G.add_edges_from([(s, sp, data) for s, sp, data in edges])
    KG_AVAILABLE = True
    logger.info("✅ Knowledge Graph ready (NetworkX)")

except ImportError:
    _G = None
    KG_AVAILABLE = False
    logger.warning("⚠️  networkx not installed, using fallback dict")

# ── Fallback dict ─────────────────────────────────────────────────────────────
_FALLBACK = {
    "chest pain":          ("Cardiology", 40),
    "shortness of breath": ("Cardiology", 30),
    "headache":            ("Neurology",  20),
    "diabetes":            ("Endocrinology", 20),
    "hypertension":        ("Cardiology", 15),
    "high blood pressure": ("Cardiology", 15),
    "fever":               ("General Physician", 15),
    "skin rash":           ("Dermatology", 10),
    "stomach pain":        ("Gastroenterology", 20),
}


# ── Public API ────────────────────────────────────────────────────────────────
def query_knowledge_graph(symptoms_text: str, history_text: str = ""):
    """
    Returns:
      - recommended_specialty: best matching specialty
      - kg_risk_score: total risk from graph edges
      - matched_nodes: list of matched symptom nodes with their edges
      - relationships: human-readable list for display
    """
    combined = (symptoms_text + " " + history_text).lower()
    specialty_scores = {}
    matched_nodes = []
    relationships = []

    if KG_AVAILABLE and _G is not None:
        for node in _G.nodes:
            if _G.nodes[node]["type"] == "symptom" and node in combined:
                for _, specialty, data in _G.out_edges(node, data=True):
                    w = data.get("weight", 10)
                    specialty_scores[specialty] = specialty_scores.get(specialty, 0) + w
                    matched_nodes.append({
                        "symptom": node,
                        "specialty": specialty,
                        "weight": w,
                        "risk": data.get("risk", "Medium"),
                    })
                    relationships.append(f"{node.title()} → {specialty} (+{w})")
    else:
        for keyword, (specialty, weight) in _FALLBACK.items():
            if keyword in combined:
                specialty_scores[specialty] = specialty_scores.get(specialty, 0) + weight
                matched_nodes.append({"symptom": keyword, "specialty": specialty, "weight": weight, "risk": "Medium"})
                relationships.append(f"{keyword.title()} → {specialty} (+{weight})")

    if not specialty_scores:
        return {
            "recommended_specialty": "General Physician",
            "kg_risk_score": 10,
            "matched_nodes": [],
            "relationships": ["No specific condition matched → General Physician"],
        }

    recommended = max(specialty_scores, key=specialty_scores.get)
    return {
        "recommended_specialty": recommended,
        "kg_risk_score": sum(specialty_scores.values()),
        "specialty_scores": specialty_scores,
        "matched_nodes": matched_nodes,
        "relationships": relationships,
    }


def get_graph_json():
    """Return graph as JSON-serialisable dict for visualisation."""
    if not KG_AVAILABLE or _G is None:
        return {"nodes": [], "edges": []}
    nodes = [{"id": n, "type": _G.nodes[n]["type"]} for n in _G.nodes]
    edges = [{"from": u, "to": v, "weight": d.get("weight", 1)} for u, v, d in _G.edges(data=True)]
    return {"nodes": nodes, "edges": edges}
