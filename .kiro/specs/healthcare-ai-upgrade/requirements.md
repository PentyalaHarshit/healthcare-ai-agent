# Requirements Document

## Introduction

This document defines requirements for a major upgrade to the existing Healthcare AI Platform — a FastAPI-based system running on port 8000 (main app) and port 8001 (MCP server), using SQLite databases (`hospital.db`, `healthcare.db`), JWT-based authentication with three roles (admin, doctor, patient), and Jinja2 frontend templates deployed via Docker Compose.

The upgrade spans seven capability areas:

1. **Vector RAG** — Replace TF-IDF keyword search with semantic vector search over medical PDFs
2. **Multi-Agent Healthcare Crew** — Upgrade from 2 agents to a full CrewAI-orchestrated 6-agent pipeline
3. **Knowledge Graph** — Store and query structured medical domain relationships
4. **Medical Image AI** — Analyse uploaded X-ray, MRI, CT, and skin images with deep learning
5. **Voice Agent** — Speech-to-text symptom intake and text-to-speech responses
6. **Real ML Risk Model** — Replace LogisticRegression with gradient-boosted models and structured features
7. **Explainable AI Dashboard** — Visual SHAP/LIME-based risk breakdown dashboard

All upgrades must integrate with the existing authentication system, preserve backward compatibility with current API endpoints, and remain deployable via Docker Compose.

---

## Glossary

- **Platform**: The Healthcare AI Platform (FastAPI application on port 8000 + MCP server on port 8001).
- **Vector_Store**: The vector database component (ChromaDB, Pinecone, or FAISS) that indexes medical document embeddings.
- **Embedding_Service**: The component that converts text into dense vector representations using sentence-transformers models.
- **RAG_Engine**: The upgraded Retrieval-Augmented Generation engine that replaces `rag_engine.py` TF-IDF logic.
- **Crew_Orchestrator**: The CrewAI-based multi-agent pipeline manager that coordinates all healthcare agents.
- **Symptom_Agent**: The CrewAI agent responsible for parsing and normalising raw patient symptom input.
- **Risk_Agent**: The CrewAI agent responsible for computing probabilistic risk scores from structured features.
- **RAG_Agent**: The CrewAI agent responsible for retrieving relevant medical guidelines from the Vector_Store.
- **Doctor_Agent**: The CrewAI agent responsible for selecting the appropriate medical specialty and recommending a doctor.
- **Appointment_Agent**: The CrewAI agent responsible for scheduling calendar appointments with urgency-aware timing.
- **Report_Agent**: The CrewAI agent responsible for compiling and formatting the final patient assessment report.
- **Knowledge_Graph**: The graph database component (Neo4j or NetworkX) storing medical entity relationships.
- **Graph_Query_Service**: The component that queries the Knowledge_Graph for symptom-to-specialty and disease-to-specialty mappings.
- **Image_Processor**: The component that pre-processes uploaded medical images (X-ray, MRI, CT, skin) before model inference.
- **Image_Classifier**: The PyTorch/MONAI deep learning model that analyses pre-processed medical images.
- **Voice_Input_Service**: The component that receives patient audio input and converts it to text using Whisper.
- **Voice_Output_Service**: The component that converts agent text responses to speech using ElevenLabs.
- **ML_Risk_Model**: The upgraded gradient-boosted risk model (XGBoost, LightGBM, or CatBoost) that replaces the existing LogisticRegression pipeline in `ml_risk_model.py`.
- **XAI_Dashboard**: The visual Explainable AI dashboard that renders SHAP/LIME-based risk breakdowns.
- **SHAP_Service**: The component that computes SHAP values for ML_Risk_Model predictions.
- **LIME_Service**: The component that generates LIME explanations for individual predictions.
- **Authenticated_User**: A user who has successfully authenticated and holds a valid JWT token.
- **Patient**: An Authenticated_User with role `patient`.
- **Doctor**: An Authenticated_User with role `doctor`.
- **Admin**: An Authenticated_User with role `admin`.
- **Risk_Score**: A floating-point probability in the range [0.0, 1.0] representing the likelihood of a high-risk health event.
- **Urgency_Level**: One of four values — `Emergency`, `Urgent`, `Soon`, or `Routine` — derived from the Risk_Score.

---

## Requirements

---

### Requirement 1: Semantic Vector Document Ingestion

**User Story:** As an Admin, I want to upload medical PDFs, guidelines, and research papers so that the system can search them semantically rather than by keyword.

#### Acceptance Criteria

1. THE Platform SHALL expose a `/admin/documents/upload` POST endpoint that accepts PDF, TXT, and DOCX files up to 50 MB in size.
2. WHEN a document file is uploaded, THE Embedding_Service SHALL chunk the document into passages of at most 512 tokens with a 64-token overlap.
3. WHEN chunking is complete, THE Embedding_Service SHALL encode each passage into a 768-dimensional dense vector using a sentence-transformers model.
4. WHEN embeddings are generated, THE Vector_Store SHALL index each passage embedding alongside its source filename and page number as metadata.
5. IF a document upload fails due to an unsupported file format, THEN THE Platform SHALL return HTTP 422 with a human-readable error message identifying the unsupported format.
6. IF a document upload fails due to exceeding the 50 MB size limit, THEN THE Platform SHALL return HTTP 413 with a message stating the size limit.
7. THE Platform SHALL restrict access to the `/admin/documents/upload` endpoint to users with the `admin` role.
8. WHEN a document is successfully indexed, THE Platform SHALL return the document identifier, chunk count, and total embedding time in milliseconds.

---

### Requirement 2: Semantic Search and RAG Retrieval

**User Story:** As a Patient, I want the system to retrieve medically relevant guidelines based on the meaning of my symptoms so that the advice is contextually accurate, not just keyword-matched.

#### Acceptance Criteria

1. WHEN a symptom query is submitted, THE RAG_Engine SHALL convert the query text into a vector using the same Embedding_Service used during ingestion.
2. WHEN the query vector is computed, THE RAG_Engine SHALL perform approximate nearest-neighbour search in the Vector_Store and retrieve the top-5 passages by cosine similarity.
3. THE RAG_Engine SHALL include only passages with a cosine similarity score above 0.45 in the retrieved context set.
4. WHEN retrieved passages are assembled, THE RAG_Engine SHALL pass them as context to the LLM along with the patient query to generate a grounded medical explanation.
5. THE RAG_Engine SHALL return each retrieved passage with its source document name, page number, and similarity score.
6. IF the Vector_Store contains no documents matching the query above the 0.45 similarity threshold, THEN THE RAG_Engine SHALL return a fallback explanation stating that no matching guidelines were found and recommend consulting a physician.
7. THE RAG_Engine SHALL replace the existing `rag_engine.py` TF-IDF implementation while maintaining the same function signature interface used by `app.py` and `mcp_server.py`.
8. FOR ALL valid query strings, encoding a query and retrieving then re-querying with the same string SHALL return the same top-ranked passages (deterministic retrieval).

---

### Requirement 3: Multi-Agent CrewAI Pipeline — Orchestration

**User Story:** As a Patient, I want my symptoms to flow through a coordinated sequence of specialised AI agents so that each step of my assessment is handled by the most appropriate agent.

#### Acceptance Criteria

1. THE Crew_Orchestrator SHALL execute agents in the following fixed sequence: Symptom_Agent → Risk_Agent → RAG_Agent → Doctor_Agent → Appointment_Agent → Report_Agent.
2. WHEN one agent completes its task, THE Crew_Orchestrator SHALL pass the full output of that agent as structured input to the next agent in the sequence.
3. IF any agent in the pipeline raises an unhandled exception, THEN THE Crew_Orchestrator SHALL halt the pipeline and return an error response identifying the failing agent and the exception message.
4. THE Crew_Orchestrator SHALL complete the full 6-agent pipeline execution within 30 seconds for a standard symptom input.
5. THE Platform SHALL expose the multi-agent pipeline via the existing `/chat` POST endpoint, maintaining backward compatibility with the current session-based request format.
6. WHEN the pipeline completes, THE Crew_Orchestrator SHALL return a structured JSON response containing the outputs of all six agents.

---

### Requirement 4: Symptom Agent

**User Story:** As a Patient, I want my free-text symptom description to be parsed into structured medical entities so that downstream agents receive clean, normalised input.

#### Acceptance Criteria

1. WHEN a free-text symptom message is received, THE Symptom_Agent SHALL extract a list of named symptom entities (e.g., "chest pain", "shortness of breath") from the input text.
2. THE Symptom_Agent SHALL normalise extracted symptom names to a canonical lowercase form defined in the medical ontology stored in the Knowledge_Graph.
3. THE Symptom_Agent SHALL extract numeric values for age, blood pressure, heart rate, and cholesterol when present in the input text.
4. IF a symptom in the input text does not match any entity in the Knowledge_Graph, THEN THE Symptom_Agent SHALL include the unmatched term in the output with an `unrecognised` flag set to `true`.
5. THE Symptom_Agent SHALL return a structured JSON object containing the symptom entity list, numeric features, and a confidence score for each extracted entity.

---

### Requirement 5: Risk Agent

**User Story:** As a Doctor, I want a probabilistic risk score computed from structured patient features so that I can triage patients more accurately than with rule-based scoring.

#### Acceptance Criteria

1. WHEN structured features from the Symptom_Agent are received, THE Risk_Agent SHALL invoke the ML_Risk_Model to compute a Risk_Score.
2. THE Risk_Agent SHALL map the Risk_Score to an Urgency_Level using the thresholds: Risk_Score ≥ 0.85 → `Emergency`, Risk_Score ≥ 0.60 → `Urgent`, Risk_Score ≥ 0.30 → `Soon`, Risk_Score < 0.30 → `Routine`.
3. THE Risk_Agent SHALL return the Risk_Score, Urgency_Level, and the raw probability vector from the ML_Risk_Model.
4. WHEN the Risk_Score is in the `Emergency` range, THE Risk_Agent SHALL include a message instructing the patient to seek emergency care immediately.
5. THE Risk_Agent SHALL pass the Urgency_Level to the Appointment_Agent for scheduling prioritisation.

---

### Requirement 6: Doctor Agent

**User Story:** As a Patient, I want the system to recommend the most appropriate medical specialty and an available doctor based on my symptoms and risk level so that I am directed to the right care.

#### Acceptance Criteria

1. WHEN symptom entities and the Risk_Score are received, THE Doctor_Agent SHALL query the Knowledge_Graph to determine the recommended medical specialty.
2. IF the Knowledge_Graph returns multiple candidate specialties, THEN THE Doctor_Agent SHALL select the specialty with the highest cumulative risk contribution.
3. THE Doctor_Agent SHALL query the existing SQLite `healthcare.db` `doctors` table to find doctors available in the recommended specialty.
4. IF no doctor is available in the recommended specialty, THEN THE Doctor_Agent SHALL fall back to General Physician and include a note in the output.
5. THE Doctor_Agent SHALL return the recommended specialty, selected doctor name, experience level, and available appointment time.

---

### Requirement 7: Report Agent

**User Story:** As a Patient, I want to receive a structured, readable health assessment report after the pipeline completes so that I understand my risk level, the reasoning behind it, and next steps.

#### Acceptance Criteria

1. WHEN all preceding agent outputs are received, THE Report_Agent SHALL compile a structured report containing: patient symptoms, Risk_Score, Urgency_Level, recommended specialty, doctor details, RAG-retrieved guidelines, and XAI risk breakdown.
2. THE Report_Agent SHALL render the report as a JSON object that maps to the existing `report.html` Jinja2 template context variables.
3. THE Report_Agent SHALL include a medical disclaimer in every report: "This is not a medical diagnosis. For emergencies, call emergency services immediately."
4. THE Report_Agent SHALL include a timestamp (ISO 8601 format) in every report.
5. IF a Patient has the `patient` role, THEN THE Platform SHALL restrict the report retrieval endpoint to that patient's own session identifier.

---

### Requirement 8: Knowledge Graph — Medical Relationships

**User Story:** As a Developer, I want medical relationships such as symptom-to-specialty and disease-to-specialty mappings stored in a graph database so that agents can traverse connections not expressible in flat SQL tables.

#### Acceptance Criteria

1. THE Knowledge_Graph SHALL store medical entities as nodes with a `type` property set to one of: `symptom`, `disease`, `specialty`, `risk_factor`.
2. THE Knowledge_Graph SHALL store directed relationships between nodes, including at minimum: `INDICATES` (symptom → disease), `REQUIRES` (disease → specialty), and `INCREASES_RISK_OF` (risk_factor → disease).
3. WHEN the Platform initialises, THE Graph_Query_Service SHALL seed the Knowledge_Graph with at minimum the following mappings: Chest Pain → Cardiology, Diabetes → Endocrinology, Hypertension → Cardiology, Shortness of Breath → Pulmonology, Headache → Neurology, Rash → Dermatology, Stomach Pain → Gastroenterology.
4. THE Graph_Query_Service SHALL expose a `get_specialty(symptom: str) -> str` interface that returns the recommended specialty for a given symptom string.
5. THE Graph_Query_Service SHALL expose a `get_related_conditions(symptom: str) -> list[str]` interface that returns diseases associated with a given symptom.
6. IF a queried symptom node does not exist in the Knowledge_Graph, THEN THE Graph_Query_Service SHALL return `"General Physician"` as the default specialty.
7. THE Knowledge_Graph implementation SHALL support both Neo4j (for production) and NetworkX (for development/testing) via a configuration toggle, without changing agent code.
8. FOR ALL symptom strings that map to a specialty in the Knowledge_Graph, querying the graph and re-querying with the same string SHALL return the same specialty (idempotent reads).

---

### Requirement 9: Medical Image Upload and Pre-processing

**User Story:** As a Patient, I want to upload X-ray, MRI, CT scan, or skin images so that the AI can analyse them and include findings in my health assessment.

#### Acceptance Criteria

1. THE Platform SHALL expose a `/patient/images/upload` POST endpoint that accepts image files with MIME types `image/jpeg`, `image/png`, and `image/dicom`.
2. WHEN an image is uploaded, THE Platform SHALL validate that the file size does not exceed 20 MB.
3. IF an uploaded file exceeds 20 MB, THEN THE Platform SHALL return HTTP 413 with a message stating the size limit.
4. IF an uploaded file has an unsupported MIME type, THEN THE Platform SHALL return HTTP 422 with a message identifying the unsupported type.
5. WHEN a valid image is received, THE Image_Processor SHALL resize the image to 224×224 pixels and normalise pixel values to the [0.0, 1.0] range before passing it to the Image_Classifier.
6. THE Platform SHALL restrict the `/patient/images/upload` endpoint to users with the `patient` or `doctor` role.

---

### Requirement 10: Medical Image Classification

**User Story:** As a Doctor, I want the system to classify uploaded medical images and return findings with confidence scores so that I can use them as a supplementary diagnostic input.

#### Acceptance Criteria

1. WHEN a pre-processed image is received, THE Image_Classifier SHALL return a predicted category from the set: `normal`, `abnormal`, or `inconclusive`.
2. THE Image_Classifier SHALL return a confidence score in the range [0.0, 1.0] for each predicted category.
3. THE Image_Classifier SHALL support image modality types: X-ray, MRI, CT Scan, and Skin Image, selected via a `modality` parameter in the upload request.
4. IF the Image_Classifier confidence for all categories is below 0.60, THEN THE Image_Classifier SHALL return `inconclusive` as the predicted category and recommend specialist review.
5. THE Image_Classifier SHALL include the image analysis result in the Report_Agent output when an image is provided.
6. THE Image_Classifier SHALL process a single uploaded image within 10 seconds on standard CPU hardware.
7. THE Platform SHALL store uploaded images in a designated directory with a unique identifier filename and SHALL NOT expose the original filename in API responses.

---

### Requirement 11: Voice Input — Speech-to-Text

**User Story:** As a Patient, I want to speak my symptoms aloud so that I do not need to type, making the system accessible to users with low digital literacy.

#### Acceptance Criteria

1. THE Platform SHALL expose a `/patient/voice/input` POST endpoint that accepts audio files in WAV, MP3, and WebM formats.
2. IF an uploaded audio file exceeds 10 MB, THEN THE Platform SHALL return HTTP 413 with a message stating the audio size limit.
3. WHEN a valid audio file is received, THE Voice_Input_Service SHALL transcribe the audio to text using Whisper and return the transcription within 15 seconds.
4. WHEN transcription is complete, THE Voice_Input_Service SHALL pass the transcribed text directly to the Symptom_Agent as a standard text symptom input.
5. THE Voice_Input_Service SHALL return the transcribed text in the API response so the Patient can verify the transcription before proceeding.
6. IF the Whisper model cannot produce a transcription (e.g., silent audio, corrupted file), THEN THE Voice_Input_Service SHALL return HTTP 422 with a message asking the patient to re-record.
7. THE Platform SHALL restrict the `/patient/voice/input` endpoint to users with the `patient` role.

---

### Requirement 12: Voice Output — Text-to-Speech

**User Story:** As a Patient, I want to hear the AI's health assessment response spoken aloud so that I can receive the information without reading from a screen.

#### Acceptance Criteria

1. WHEN a Report_Agent output is generated, THE Voice_Output_Service SHALL convert the report summary text to speech using ElevenLabs and return an audio file in MP3 format.
2. THE Voice_Output_Service SHALL complete text-to-speech conversion within 10 seconds for report summaries up to 500 words.
3. IF the ElevenLabs API is unavailable, THEN THE Voice_Output_Service SHALL log the error and return the text report without audio, including a message noting that audio is temporarily unavailable.
4. THE Platform SHALL expose a `/patient/voice/output` GET endpoint that streams the MP3 audio response for a given session identifier.
5. THE Platform SHALL restrict the `/patient/voice/output` endpoint to users with the `patient` role.
6. THE Voice_Output_Service SHALL NOT include the medical disclaimer in the synthesised audio to comply with safe messaging guidelines; the disclaimer SHALL appear only in the text report.

---

### Requirement 13: Gradient-Boosted ML Risk Model

**User Story:** As a Developer, I want to replace the existing LogisticRegression text classifier with a structured gradient-boosted model so that risk scoring uses clinical features with documented importance.

#### Acceptance Criteria

1. THE ML_Risk_Model SHALL accept the following structured input features: age (integer), systolic blood pressure (integer), diastolic blood pressure (integer), heart rate (integer), cholesterol (float), diabetes flag (boolean), and a symptom severity score (float in [0.0, 1.0]).
2. THE ML_Risk_Model SHALL be trained using one of XGBoost, LightGBM, or CatBoost with hyperparameters stored in a versioned configuration file.
3. THE ML_Risk_Model SHALL output a Risk_Score as a floating-point probability in the range [0.0, 1.0] and a class label of `High`, `Medium`, or `Low`.
4. THE ML_Risk_Model SHALL achieve a minimum cross-validated AUC-ROC of 0.75 on the training dataset before deployment.
5. THE ML_Risk_Model SHALL be serialised to a versioned model file (e.g., `risk_model_v2.pkl`) separate from the existing `risk_model.pkl` to allow rollback.
6. IF the ML_Risk_Model receives a feature vector with missing values, THEN THE ML_Risk_Model SHALL apply median imputation using training-set medians stored alongside the model file.
7. THE Platform SHALL expose a `/admin/model/retrain` POST endpoint restricted to the `admin` role that triggers model retraining with a new dataset file.
8. WHEN retraining completes, THE Platform SHALL return the new model's AUC-ROC score and a model version identifier.

---

### Requirement 14: SHAP-Based Explainability

**User Story:** As a Doctor, I want to see which features contributed most to a patient's risk score so that I can validate the model's reasoning and explain it to the patient.

#### Acceptance Criteria

1. WHEN a Risk_Score is computed, THE SHAP_Service SHALL compute SHAP values for each input feature using the trained ML_Risk_Model.
2. THE SHAP_Service SHALL return a feature contribution object mapping each feature name to its SHAP value, sorted in descending order of absolute value.
3. THE SHAP_Service SHALL complete SHAP computation within 2 seconds per prediction on standard CPU hardware.
4. THE XAI_Dashboard SHALL display each feature contribution as a horizontal bar chart with the feature name, SHAP value, and direction (risk-increasing vs. risk-decreasing) colour-coded.
5. THE XAI_Dashboard SHALL display the overall Risk_Score as a percentage gauge at the top of the dashboard.
6. THE XAI_Dashboard SHALL be accessible via the existing `doctor.html` and `patient.html` templates without requiring a separate page load.
7. FOR ALL valid feature vectors, the sum of all SHAP values plus the model's expected base value SHALL equal the model's raw log-odds prediction output (SHAP additivity property).

---

### Requirement 15: LIME-Based Explainability

**User Story:** As a Patient, I want a simple, plain-language explanation of why I received my risk score so that I can understand it without medical training.

#### Acceptance Criteria

1. WHEN a Risk_Score is computed, THE LIME_Service SHALL generate a LIME explanation identifying the top-5 features that most influenced the prediction.
2. THE LIME_Service SHALL produce LIME explanations using 1 000 neighbourhood samples per prediction.
3. THE LIME_Service SHALL complete LIME explanation generation within 5 seconds per prediction on standard CPU hardware.
4. THE XAI_Dashboard SHALL render the LIME explanation as a ranked list of feature names with their contribution direction and magnitude, displayed in plain language (e.g., "Chest Pain increased your risk by 40%").
5. IF the ML_Risk_Model is retrained, THEN THE LIME_Service SHALL use the new model without requiring a Platform restart.

---

### Requirement 16: XAI Dashboard Integration

**User Story:** As an Admin, I want a unified explainability dashboard accessible to doctors and patients so that all model explanations are visible in one place.

#### Acceptance Criteria

1. THE XAI_Dashboard SHALL be rendered server-side and served via the existing Jinja2 templating system at the `/dashboard` route.
2. THE Platform SHALL restrict the full SHAP feature-level detail view to users with the `doctor` or `admin` role.
3. THE Platform SHALL restrict the LIME plain-language summary view to users with the `patient`, `doctor`, or `admin` role.
4. THE XAI_Dashboard SHALL load all chart data via a `/api/xai/explain` GET endpoint that returns JSON containing both SHAP and LIME outputs for a given session identifier.
5. WHEN a user navigates to the `/dashboard` route without an active session, THE Platform SHALL redirect the user to the `/login-page` route.
6. THE XAI_Dashboard SHALL render correctly on viewport widths from 375 px (mobile) to 1920 px (desktop) using responsive CSS.

---

### Requirement 17: Backward Compatibility and Integration

**User Story:** As a Developer, I want all existing API endpoints and authentication flows to continue working after the upgrade so that existing integrations are not broken.

#### Acceptance Criteria

1. THE Platform SHALL preserve the request and response schema of the existing `/chat`, `/book`, `/login`, `/analyze`, and `/hospital` endpoints.
2. WHEN the upgraded Platform starts, THE Platform SHALL initialise the SQLite `hospital.db` and `healthcare.db` databases using the same `init_db()` logic as the current implementation.
3. THE Platform SHALL continue to support JWT authentication using the `Authorization: Bearer <token>` header and the `/login` OAuth2 token endpoint.
4. THE Platform SHALL continue to enforce role-based access control for `admin`, `doctor`, and `patient` roles on all existing protected endpoints.
5. THE Crew_Orchestrator SHALL expose a feature flag `USE_CREW_PIPELINE` in environment configuration; WHEN set to `false`, THE Platform SHALL fall back to the original single-agent logic in `main.py`.
6. WHERE Docker Compose is used as the deployment environment, THE Platform SHALL add new service definitions (e.g., vector store, Neo4j) as additional services without modifying the existing `mcp_server` and `main_app` service definitions.

---

### Requirement 18: Performance and Reliability

**User Story:** As an Admin, I want the upgraded platform to handle concurrent patient requests reliably so that system performance does not degrade under load.

#### Acceptance Criteria

1. THE Platform SHALL handle at least 20 concurrent `/chat` requests without returning HTTP 5xx errors.
2. WHEN the Vector_Store is unavailable, THE RAG_Engine SHALL fall back to the existing TF-IDF retrieval logic and log a warning with severity `WARNING`.
3. WHEN the Knowledge_Graph is unavailable, THE Graph_Query_Service SHALL fall back to the hard-coded specialty mapping rules in the existing `database.py` `HealthRule` table and log a warning.
4. WHEN the ML_Risk_Model file is missing or corrupt, THE Platform SHALL retrain the model from the embedded demo dataset and log an error with severity `ERROR`.
5. THE Platform SHALL return HTTP 503 with a `Retry-After` header of 30 seconds when all fallback mechanisms are exhausted for a critical service dependency.
6. THE Platform SHALL log all agent pipeline executions with a unique trace identifier, agent name, input hash, output hash, and execution duration in milliseconds.
