def combined_uncertainty_agent(xai_analysis, bayesian_risk, ml_risk):
    """Combines uncertainty from XAI, Bayesian, and ML models."""
    xai_confidence = xai_analysis.get("confidence", 0)
    bayesian_confidence = 1 - bayesian_risk.get("posterior_risk", 0)
    ml_confidence = ml_risk.get("confidence", 0)

    # Average confidence across all three agents
    combined_confidence = (xai_confidence + bayesian_confidence + ml_confidence) / 3

    # Determine treatment recommendation based on combined confidence
    if combined_confidence >= 0.8:
        treatment = "High confidence diagnosis - proceed with standard protocol"
    elif combined_confidence >= 0.6:
        treatment = "Moderate confidence - recommend secondary confirmation"
    else:
        treatment = "Low confidence - recommend specialist consultation"

    return {
        "xai_confidence": round(xai_confidence, 2),
        "bayesian_confidence": round(bayesian_confidence, 2),
        "ml_confidence": round(ml_confidence, 2),
        "combined_confidence": round(combined_confidence, 2),
        "treatment_recommendation": treatment,
        "uncertainty_level": "Low" if combined_confidence >= 0.8 else ("Medium" if combined_confidence >= 0.6 else "High")
    }
