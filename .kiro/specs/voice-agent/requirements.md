# Requirements Document

## Introduction

The Voice Agent feature extends the existing Healthcare AI platform to support spoken interaction.
Patients can speak their symptoms aloud, have their speech transcribed by OpenAI Whisper (STT),
route the transcription through the existing multi-turn healthcare agent pipeline at `POST /chat`,
and receive the response read back to them via ElevenLabs text-to-speech (TTS). The feature is
exposed as a new FastAPI endpoint and as an embedded browser widget on the existing patient page.

The pipeline is: **Microphone → Whisper STT → Healthcare Agent (`/chat`) → ElevenLabs TTS → Speaker/Audio output**.

---

## Glossary

- **Voice_Agent**: The end-to-end voice interaction system comprising the STT, routing, and TTS stages.
- **STT_Service**: The OpenAI Whisper-based speech-to-text component that converts raw audio into text.
- **TTS_Service**: The ElevenLabs text-to-speech component that converts agent text replies into audio.
- **Healthcare_Agent**: The existing multi-turn chat agent accessible at `POST /chat` in `main.py`.
- **Voice_Endpoint**: The new FastAPI route `POST /voice-chat` that accepts audio input and returns audio output.
- **Voice_Widget**: The browser-based UI component embedded in the patient page (`patient.html`) that records audio and plays back the TTS response.
- **Session**: An in-memory stateful conversation context managed by the Healthcare_Agent, identified by a `session_id`.
- **Audio_Payload**: A binary audio file (WAV or WebM format) uploaded by the patient or captured by the Voice_Widget.
- **Transcript**: The plain-text string produced by the STT_Service from an Audio_Payload.
- **Agent_Reply**: The plain-text string returned by the Healthcare_Agent for a given Transcript and Session.
- **Audio_Response**: The binary audio file produced by the TTS_Service from an Agent_Reply.

---

## Requirements

### Requirement 1: Speech-to-Text Transcription

**User Story:** As a patient, I want to speak my symptoms rather than type them, so that I can interact with the healthcare assistant hands-free and naturally.

#### Acceptance Criteria

1. WHEN an Audio_Payload is received at the Voice_Endpoint, THE STT_Service SHALL transcribe the audio into a Transcript using the OpenAI Whisper API.
2. WHEN the Audio_Payload is in WAV or WebM format and does not exceed 25 MB, THE STT_Service SHALL return a non-empty Transcript within 10 seconds.
3. IF the Audio_Payload is empty or malformed, THEN THE STT_Service SHALL return an error response with HTTP status 400 and a descriptive error message.
4. IF the Audio_Payload exceeds 25 MB, THEN THE Voice_Endpoint SHALL reject the request with HTTP status 413 and a descriptive error message before calling the STT_Service.
5. IF the STT_Service API call fails or times out after 10 seconds, THEN THE Voice_Endpoint SHALL return an error response with HTTP status 502 and a message indicating the transcription service is unavailable.
6. THE STT_Service SHALL accept audio in English language by default.

---

### Requirement 2: Healthcare Agent Routing

**User Story:** As a patient, I want my spoken symptoms to be processed by the same intelligent healthcare agent that handles text chat, so that I receive the same quality of risk assessment and doctor recommendations regardless of input modality.

#### Acceptance Criteria

1. WHEN a Transcript is produced by the STT_Service, THE Voice_Agent SHALL forward the Transcript as the `message` field to the Healthcare_Agent via the existing `POST /chat` logic.
2. WHEN an existing `session_id` is provided in the voice request, THE Voice_Agent SHALL include that `session_id` in the Healthcare_Agent call to continue the multi-turn Session.
3. WHEN no `session_id` is provided in the voice request, THE Voice_Agent SHALL omit `session_id` from the Healthcare_Agent call so that the Healthcare_Agent creates a new Session.
4. THE Voice_Agent SHALL preserve the `session_id` returned by the Healthcare_Agent and include it in the Voice_Endpoint response so the Voice_Widget can maintain Session continuity across turns.
5. IF the Healthcare_Agent returns an error response, THEN THE Voice_Agent SHALL propagate the error to the caller with the original HTTP status code and error detail.
6. THE Voice_Agent SHALL pass the Transcript to the Healthcare_Agent without modification or filtering.

---

### Requirement 3: Text-to-Speech Response

**User Story:** As a patient, I want the healthcare assistant's reply to be spoken aloud, so that I can hear the response without reading a screen.

#### Acceptance Criteria

1. WHEN an Agent_Reply is received from the Healthcare_Agent, THE TTS_Service SHALL synthesize the Agent_Reply into an Audio_Response using the ElevenLabs API.
2. THE TTS_Service SHALL produce Audio_Response in MP3 format.
3. WHEN the Agent_Reply does not exceed 5000 characters, THE TTS_Service SHALL return an Audio_Response within 8 seconds.
4. IF the Agent_Reply exceeds 5000 characters, THEN THE TTS_Service SHALL truncate the text to 5000 characters before synthesis and append a note indicating the response was truncated.
5. IF the TTS_Service API call fails or times out after 8 seconds, THEN THE Voice_Endpoint SHALL return the Agent_Reply as plain text with HTTP status 200 and a header `X-TTS-Fallback: true`, so the patient can still read the response.
6. THE TTS_Service SHALL use a consistent voice identifier configured via the `ELEVENLABS_VOICE_ID` environment variable.

---

### Requirement 4: Voice Chat API Endpoint

**User Story:** As a developer or system integrator, I want a single API endpoint that accepts audio input and returns audio output, so that voice interaction can be integrated into any client application.

#### Acceptance Criteria

1. THE Voice_Endpoint SHALL be accessible at `POST /voice-chat` on the FastAPI application running on port 8000.
2. THE Voice_Endpoint SHALL accept a multipart form request containing an `audio` file field and an optional `session_id` string field.
3. WHEN the request is valid, THE Voice_Endpoint SHALL return a response containing the MP3 Audio_Response as a binary stream with `Content-Type: audio/mpeg`.
4. THE Voice_Endpoint SHALL include a `X-Session-Id` response header containing the active `session_id` so clients can continue multi-turn sessions.
5. THE Voice_Endpoint SHALL include a `X-Transcript` response header containing the URL-encoded Transcript so clients can display what was heard.
6. IF the `OPENAI_API_KEY` environment variable is not set at startup, THEN THE Voice_Endpoint SHALL return HTTP status 503 on every request with a message indicating the STT service is misconfigured.
7. IF the `ELEVENLABS_API_KEY` environment variable is not set at startup, THEN THE Voice_Endpoint SHALL return HTTP status 503 on every request with a message indicating the TTS service is misconfigured.
8. THE Voice_Endpoint SHALL log the Transcript and the `session_id` for each request to the application log at INFO level.

---

### Requirement 5: Browser-Based Voice Widget

**User Story:** As a patient using the web application, I want a voice button on the patient chat page, so that I can start and stop recording and hear the assistant's response without leaving the page.

#### Acceptance Criteria

1. THE Voice_Widget SHALL be embedded in the existing patient page (`/patient-page`) as a microphone button alongside the existing text input and send button.
2. WHEN the patient clicks the microphone button, THE Voice_Widget SHALL request browser microphone permission and begin recording audio.
3. WHILE recording is active, THE Voice_Widget SHALL display a visual indicator (pulsing or animated) to show that the microphone is capturing audio.
4. WHEN the patient clicks the microphone button a second time, THE Voice_Widget SHALL stop recording and submit the captured audio to the Voice_Endpoint.
5. WHILE the Voice_Endpoint request is in progress, THE Voice_Widget SHALL display a loading indicator and disable the microphone button to prevent duplicate submissions.
6. WHEN the Voice_Endpoint returns an Audio_Response, THE Voice_Widget SHALL play the Audio_Response through the browser's audio output automatically.
7. WHEN the Voice_Endpoint returns a Transcript via the `X-Transcript` response header, THE Voice_Widget SHALL display the Transcript as a patient message bubble in the existing chat box, consistent with existing text message display.
8. WHEN the Voice_Endpoint returns an Agent_Reply readable from the response, THE Voice_Widget SHALL display the Agent_Reply as an agent message bubble in the existing chat box.
9. THE Voice_Widget SHALL persist the `session_id` from the `X-Session-Id` response header in browser memory and include it in all subsequent voice and text requests for that page session.
10. IF the browser does not support the MediaRecorder API, THEN THE Voice_Widget SHALL hide the microphone button and display a tooltip message: "Voice input is not supported in this browser."
11. IF microphone permission is denied by the patient, THEN THE Voice_Widget SHALL display an inline error message: "Microphone access was denied. Please allow microphone access in your browser settings."
12. IF the Voice_Endpoint returns an error, THEN THE Voice_Widget SHALL display a human-readable error message in the chat box and re-enable the microphone button.

---

### Requirement 6: Security and Authentication

**User Story:** As a system administrator, I want the voice endpoint to respect the same security boundaries as the existing application, so that patient voice data is handled safely.

#### Acceptance Criteria

1. THE Voice_Endpoint SHALL not store or persist Audio_Payloads to disk or any database after the request completes.
2. THE Voice_Endpoint SHALL not store or persist Audio_Responses to disk or any database after the response is sent.
3. THE Voice_Endpoint SHALL transmit Audio_Payloads to the STT_Service exclusively over HTTPS.
4. THE Voice_Endpoint SHALL transmit Agent_Reply text to the TTS_Service exclusively over HTTPS.
5. THE Voice_Endpoint SHALL read API keys exclusively from environment variables (`OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`) and SHALL NOT accept API keys in request parameters or headers.

---

### Requirement 7: Configuration and Deployment

**User Story:** As a DevOps engineer, I want the voice agent dependencies to be manageable through the existing Docker Compose setup, so that deployment requires no manual steps beyond environment variable configuration.

#### Acceptance Criteria

1. THE Voice_Agent SHALL declare `openai` and `elevenlabs` as Python package dependencies in `requirements.txt` with pinned version numbers.
2. WHEN the Docker Compose environment is started, THE Voice_Agent SHALL be available within the existing FastAPI container on port 8000 without a separate service container.
3. THE Voice_Agent SHALL read all external service credentials from environment variables injected at container startup.
4. WHERE the `ELEVENLABS_VOICE_ID` environment variable is not set, THE TTS_Service SHALL use a default voice identifier value of `"Rachel"` so the service can start without explicit voice configuration.
