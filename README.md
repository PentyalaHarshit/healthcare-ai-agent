# 🏥 Healthcare AI Agent

> A multi-turn, multi-agent AI system for symptom analysis, risk prediction, and intelligent doctor appointment booking — built with FastAPI, explainable AI (XAI), RAG, and Docker.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [API Reference](#api-reference)
- [User Roles & Credentials](#user-roles--credentials)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Disclaimer](#disclaimer)

---

## Overview

**Healthcare AI Agent** is a conversational, multi-agent healthcare assistant that helps patients:

1. Describe their symptoms through a guided multi-turn chat
2. Receive an AI-driven **risk score** (0–100) with **explainable AI (XAI)** reasoning
3. Get **RAG-powered medical context** retrieved from a curated knowledge base
4. Find and **book available doctors** matched by specialty and hospital
5. Receive a **priority classification** (Normal / High / Emergency)

The system uses a **Router Agent** to classify incoming queries and route them to the appropriate agent pipeline.

---

## Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Turn Chat** | Guided conversation flow: hospital → symptoms → analysis |
| 🧠 **Risk Prediction** | Symptom-based risk scoring (0–100) |
| 🔍 **XAI Explanations** | Transparent, point-by-point breakdown of risk factors |
| 📚 **RAG Medical Context** | Keyword-matched retrieval from a medical knowledge base |
| 🏥 **Smart Doctor Matching** | Matches patients to the right specialty and available slots |
| 📅 **Appointment Booking** | Real-time slot reservation with fallback to alternate hospitals |
| 🔐 **JWT Auth** | Role-based access control (Patient / Doctor / Admin) |
| 🐳 **Docker Support** | Fully containerized with Docker Compose (MCP + Main App) |
| 🗓️ **Google Calendar** | Google Calendar integration for appointment scheduling |
| ⚕️ **MCP Server** | Dedicated Model Context Protocol server for tool execution |

---

## Architecture

```
User
 │
 ▼
┌─────────────────────────────────────────┐
│            Router Agent                 │  ← Classifies: healthcare / general
└────────────────┬────────────────────────┘
                 │
        ┌────────▼────────┐
        │ Healthcare Agent │
        └────────┬────────┘
                 │
     ┌───────────┼──────────────┐
     ▼           ▼              ▼
Risk Model    XAI Agent      RAG Agent
(score 0-100) (explanations) (medical context)
     │
     ▼
Doctor Selection Agent
(specialty → available slots)
     │
     ▼
Booking → SQLite DB
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) |
| **Server** | Uvicorn (ASGI) |
| **Database** | SQLite + SQLAlchemy |
| **Auth** | JWT (python-jose) + passlib |
| **ML** | scikit-learn, pandas, joblib |
| **Templates** | Jinja2 |
| **AI/RAG** | LangChain Core |
| **Calendar** | Google Calendar API |
| **Container** | Docker + Docker Compose |
| **Language** | Python 3.11 |

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- Docker & Docker Compose *(optional, for containerized setup)*

---

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/PentyalaHarshit/healthcare-ai-agent.git
cd healthcare-ai-agent

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the MCP Server (in a separate terminal)
uvicorn mcp_server:app --host 0.0.0.0 --port 8001

# 5. Start the Main App
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit: [http://localhost:8000](http://localhost:8000)

---

### Docker Setup

```bash
# Build and start both services
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop services
docker-compose down
```

| Service | URL |
|---|---|
| Main App | http://localhost:8000 |
| MCP Server | http://localhost:8001 |

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/login` | Login and receive JWT token |

### Chat / Agent

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Multi-turn healthcare chat |
| `POST` | `/analyze` | Analyze symptoms (form-based) |
| `POST` | `/book` | Book a doctor appointment slot |

### Pages

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home page |
| `GET` | `/login-page` | Login page |
| `GET` | `/patient-page` | Patient dashboard |
| `GET` | `/doctor-page` | Doctor dashboard |
| `GET` | `/admin-page` | Admin dashboard |
| `GET` | `/report-page` | Report page |

### MCP Server Tools

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/tools/find_doctors?specialty=Cardiology` | Find available doctors by specialty |
| `GET` | `/tools/health_rules` | Get health rules knowledge base |

---

### Chat API Example

**Start a new session:**
```json
POST /chat
{
  "message": "I have chest pain and diabetes"
}
```

**Response:**
```json
{
  "session_id": "abc-123",
  "router_agent": "Healthcare Agent selected",
  "reply": "What hospital would you like to visit?"
}
```

**Continue the session:**
```json
POST /chat
{
  "session_id": "abc-123",
  "message": "Texas Health Frisco"
}
```

---

## User Roles & Credentials

> ⚠️ These are demo credentials for development only. Change them before production use.

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |
| Doctor | `doctor` | `doctor123` |
| Patient | `patient` | `patient123` |

---

## Project Structure

```
healthcare-ai-agent/
├── main.py                 # FastAPI app, all agents & endpoints
├── mcp_server.py           # MCP (Model Context Protocol) server
├── auth.py                 # JWT authentication & role-based access
├── database.py             # SQLAlchemy models & DB initialization
├── appointment_agent.py    # Appointment booking logic
├── bayesian_agent.py       # Bayesian risk agent
├── uncertainty_agent.py    # Uncertainty quantification agent
├── xai_agent.py            # Explainable AI agent
├── rag_engine.py           # RAG retrieval engine
├── ml_risk_model.py        # ML-based risk model
├── google_calendar.py      # Google Calendar integration
├── risk_model.pkl          # Trained risk model artifact
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Multi-service orchestration
├── templates/              # Jinja2 HTML templates
│   ├── index.html
│   ├── patient.html
│   ├── doctor.html
│   ├── admin.html
│   ├── report.html
│   └── ...
├── static/                 # Static assets (CSS, JS)
│   └── app.js
└── medical_guidelines/     # Medical knowledge base files
```

---

## How It Works

### 1. Router Agent
Classifies incoming messages as `healthcare` or `general` based on keyword matching (e.g., "chest pain", "doctor", "hospital").

### 2. Risk Prediction
Scores symptoms on a 0–100 scale:

| Symptom | Risk Points |
|---|---|
| Chest Pain | +40 |
| Shortness of Breath | +30 |
| Diabetes | +25 |
| High Blood Pressure | +20 |
| Fever | +15 |
| Age ≥ 60 | +10 |

### 3. Priority Classification

| Score | Priority |
|---|---|
| ≥ 90 | 🔴 Emergency |
| 60–89 | 🟡 High |
| < 60 | 🟢 Normal |

### 4. Specialty Routing

| Symptoms | Specialty |
|---|---|
| Chest pain / Diabetes | Cardiology |
| Breathing / Shortness of breath | Pulmonology |
| Other | General |

### 5. XAI Explanations
Each risk factor is explained individually, providing full transparency into why a risk score was assigned — no black box.

### 6. RAG Medical Context
Top-3 relevant medical facts are retrieved from the knowledge base using keyword overlap scoring and presented alongside the risk report.

---

## Disclaimer

> ⚠️ **This system is not a substitute for professional medical advice, diagnosis, or treatment.** Always seek the advice of your physician or a qualified health provider with any questions you may have regarding a medical condition. In case of emergency, call your local emergency services immediately.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">Built with ❤️ using FastAPI, Python, and AI</p>
