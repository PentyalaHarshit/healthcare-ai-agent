def bayesian_risk_agent(matched_rules):
    """Bayesian-style risk computation agent."""
    prior_risk = 0.10
    risk_multiplier = 1.0

    for rule in matched_rules:
        risk_multiplier *= rule.get("multiplier", 1)

    posterior_risk = prior_risk * risk_multiplier
    posterior_risk = min(posterior_risk, 0.95)

    return {
        "prior_risk": prior_risk,
        "posterior_risk": round(posterior_risk, 2),
        "risk_percentage": round(posterior_risk * 100, 2),
        "method": "Bayesian-style risk update"
    }
