const API_URL = "";

// Safe storage fallback
const safeStorage = {
    getItem(key) {
        try { return localStorage.getItem(key); } catch (e) { return this[key] || null; }
    },
    setItem(key, val) {
        try { localStorage.setItem(key, val); } catch (e) { this[key] = val; }
    },
    removeItem(key) {
        try { localStorage.removeItem(key); } catch (e) { delete this[key]; }
    },
    clear() {
        try { localStorage.clear(); } catch (e) {
            for (let k in this) {
                if (typeof this[k] !== 'function') delete this[k];
            }
        }
    }
};

function saveToken(token, role) {
    safeStorage.setItem("token", token);
    safeStorage.setItem("role", role);
}

function getToken() {
    return safeStorage.getItem("token");
}

function getSessionId() {
    return safeStorage.getItem("chat_session_id") || null;
}

function saveSessionId(id) {
    if (id) {
        safeStorage.setItem("chat_session_id", id);
    } else {
        safeStorage.removeItem("chat_session_id");
    }
}

async function login() {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);

    const response = await fetch(`${API_URL}/login`, {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    if (!response.ok) {
        alert("Login failed");
        return;
    }

    saveToken(data.access_token, data.role);

    if (data.role === "admin") {
        window.location.href = "/admin-page";
    } else if (data.role === "doctor") {
        window.location.href = "/doctor-page";
    } else {
        window.location.href = "/patient-page";
    }
}

function logout() {
    localStorage.clear();
    window.location.href = "/login-page";
}

// -------------------------------------------------------------
// Chat and Multi-Turn Agent Logic
// -------------------------------------------------------------

function appendMessage(sender, text, isHtml = false) {
    const chatBox = document.getElementById("chat-box");
    if (!chatBox) return;

    const messageDiv = document.createElement("div");
    messageDiv.className = `chat-message ${sender} animate-fade-in`;

    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "bubble";
    
    if (isHtml) {
        bubbleDiv.innerHTML = text;
    } else {
        bubbleDiv.textContent = text;
    }

    messageDiv.appendChild(bubbleDiv);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const inputElement = document.getElementById("chat-input");
    const sendBtn = document.querySelector(".btn-send");
    const message = inputElement.value.trim();
    
    if (!message) return;

    // Clear input
    inputElement.value = "";
    inputElement.disabled = true;
    if (sendBtn) sendBtn.disabled = true;

    // Append user message to chat
    appendMessage("user", message);

    try {
        const sessionId = getSessionId();
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: sessionId,
                message: message
            })
        });

        const data = await response.json();

        if (data.error) {
            appendMessage("agent", `Error: ${data.error}`);
            saveSessionId(null); // Clear invalid session
            return;
        }

        // Save session ID
        saveSessionId(data.session_id);

        if (data.reply) {
            // Standard text reply
            appendMessage("agent", data.reply, true);
        } else if (data.risk_score !== undefined) {
            // Rich analysis and recommendations card
            renderAnalysisCard(data);
        }

    } catch (err) {
        console.error(err);
        appendMessage("agent", "Error connecting to server. Please try again.");
    } finally {
        inputElement.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        inputElement.focus();
    }
}

function renderAnalysisCard(data) {
    const chatBox = document.getElementById("chat-box");
    if (!chatBox) return;

    // Map priority classes and names
    let priorityClass = "normal";
    let priorityText = "NORMAL";
    if (data.priority === "Emergency") {
        priorityClass = "emergency";
        priorityText = "EMERGENCY";
    } else if (data.priority === "High") {
        priorityClass = "high";
        priorityText = "HIGH";
    }

    let doctorsHtml = "";
    if (data.available_doctors && data.available_doctors.length > 0) {
        doctorsHtml = `
            <div class="doctor-slots-container">
                <h4>Select & Book Available Doctor:</h4>
                <div class="slots-grid">
                    ${data.available_doctors.map(d => `
                        <div class="slot-card" id="slot-card-${d.slot_id}">
                            <div class="slot-doc"><b>${d.doctor_name}</b> (${d.specialty})</div>
                            <div class="slot-time">📅 ${d.date} at ⏰ ${d.time}</div>
                            <button class="btn-book-slot" onclick="bookSlot(${d.slot_id}, this)">Book Slot</button>
                        </div>
                    `).join("")}
                </div>
            </div>
        `;
    } else {
        let suggestedHtml = "";
        if (data.suggested_hospitals && data.suggested_hospitals.length > 0) {
            suggestedHtml = `
                <p>However, slots for <b>${data.recommended_specialty}</b> are available at other hospitals:</p>
                <ul>
                    ${data.suggested_hospitals.map(h => `<li><b>${h}</b></li>`).join("")}
                </ul>
                <p class="suggestion-tip">🔄 Reset the chat and try booking at one of these locations!</p>
            `;
        } else {
            suggestedHtml = `<p>No available slots found for <b>${data.recommended_specialty}</b> at other locations.</p>`;
        }
        
        doctorsHtml = `
            <div class="no-doctors-container">
                <p class="warning-text">⚠️ No available doctors found at <b>${data.hospital}</b> for <b>${data.recommended_specialty}</b>.</p>
                ${suggestedHtml}
            </div>
        `;
    }

    let xaiHtml = "";
    if (data.xai_details && data.xai_details.length > 0) {
        xaiHtml = `
            <div class="report-section collapsible">
                <h4>Why Risk Is High?</h4>
                <ul style="list-style: none; padding-left: 0; margin-top: 10px;">
                    ${data.xai_details.map(item => `
                        <li style="display: flex; justify-content: space-between; font-family: monospace; font-size: 1.05rem; color: #cbd5e1; margin-bottom: 8px;">
                            <span>✓ ${item.name}</span>
                            <span>+${item.points}</span>
                        </li>
                    `).join("")}
                </ul>
                <div style="border-top: 1px solid var(--border-color); margin-top: 12px; padding-top: 12px; font-weight: bold; font-size: 1.1rem; display: flex; justify-content: space-between; color: #cbd5e1;">
                    <span>Total Risk Score</span>
                    <span>= ${data.risk_score}</span>
                </div>
            </div>
        `;
    }

    const cardContent = `
        <div class="analysis-report card animate-slide-up">
            <div class="report-header">
                <h3>📋 Medical Assessment Report</h3>
                <div class="risk ${priorityClass}">${priorityText}</div>
            </div>
            
            <div class="report-section">
                <h3>Cardiovascular Risk Score</h3>
                <div class="progress" style="margin-top: 10px;">
                    <div class="bar" style="width: ${data.risk_score}%">
                        ${data.risk_score}%
                    </div>
                </div>
            </div>

            <!-- XAI Section -->
            ${xaiHtml}

            <div class="report-section">
                <p class="recommendation"><b>Recommendation:</b> ${data.recommendation}</p>
            </div>

            <div class="report-section collapsible">
                <h4>📚 RAG Medical Context & Guidelines:</h4>
                <ul>
                    ${data.rag_medical_context.map(doc => `<li>${doc}</li>`).join("")}
                </ul>
            </div>

            ${doctorsHtml}

            <div class="report-section disclaimer">
                <small>⚠️ <i>Disclaimer: ${data.disclaimer}</i></small>
            </div>
        </div>
    `;

    const messageDiv = document.createElement("div");
    messageDiv.className = "chat-message agent";
    messageDiv.innerHTML = cardContent;

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function bookSlot(slotId, buttonElement) {
    buttonElement.disabled = true;
    buttonElement.textContent = "Booking...";

    try {
        const response = await fetch(`${API_URL}/book`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                slot_id: slotId
            })
        });

        const data = await response.json();

        if (data.status === "success") {
            buttonElement.className = "btn-book-slot booked";
            buttonElement.textContent = "Confirmed ✓";
            
            const card = document.getElementById(`slot-card-${slotId}`);
            if (card) {
                card.classList.add("booked-card");
            }
            
            appendMessage("agent", "🎉 Appointment booked successfully! We have confirmed your slot.");
        } else {
            buttonElement.disabled = false;
            buttonElement.textContent = "Book Slot";
            alert(`Booking failed: ${data.message}`);
        }
    } catch (err) {
        console.error(err);
        buttonElement.disabled = false;
        buttonElement.textContent = "Book Slot";
        alert("Error booking appointment. Please try again.");
    }
}

function resetSession() {
    saveSessionId(null);
    const chatBox = document.getElementById("chat-box");
    if (chatBox) {
        chatBox.innerHTML = `
            <div class="chat-message agent animate-fade-in">
                <div class="bubble">
                    Chat session has been reset. How can I help you today? (Try typing: <i>"I need medical help"</i>)
                </div>
            </div>
        `;
    }
    const inputElement = document.getElementById("chat-input");
    if (inputElement) {
        inputElement.value = "";
        inputElement.focus();
    }
}

// Auto-focus input on page load
window.addEventListener("DOMContentLoaded", () => {
    const inputElement = document.getElementById("chat-input");
    if (inputElement) inputElement.focus();
});


// =============================================================================
// Voice Input — Browser MediaRecorder → /patient/voice/input
// =============================================================================

let _mediaRecorder = null;
let _audioChunks = [];
let _isRecording = false;

async function toggleVoiceInput() {
    if (_isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Your browser does not support audio recording. Please type your symptoms.");
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        _audioChunks = [];
        _mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

        _mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) _audioChunks.push(e.data);
        };

        _mediaRecorder.onstop = async () => {
            const blob = new Blob(_audioChunks, { type: "audio/webm" });
            stream.getTracks().forEach(t => t.stop());
            await uploadAudioForTranscription(blob);
        };

        _mediaRecorder.start();
        _isRecording = true;

        const voiceBtn = document.getElementById("voice-btn");
        const voiceStatus = document.getElementById("voice-status");
        if (voiceBtn) voiceBtn.textContent = "⏹️";
        if (voiceStatus) voiceStatus.style.display = "block";

    } catch (err) {
        alert("Could not access microphone: " + err.message);
    }
}

function stopRecording() {
    if (_mediaRecorder && _isRecording) {
        _mediaRecorder.stop();
        _isRecording = false;
        const voiceBtn = document.getElementById("voice-btn");
        const voiceStatus = document.getElementById("voice-status");
        if (voiceBtn) voiceBtn.textContent = "🎤";
        if (voiceStatus) voiceStatus.style.display = "none";
        appendMessage("agent", "⏳ Transcribing your voice… please wait.");
    }
}

async function uploadAudioForTranscription(blob) {
    const token = getToken();
    if (!token) {
        appendMessage("agent", "⚠️ Please log in to use voice input.");
        return;
    }

    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");

    try {
        const resp = await fetch("/patient/voice/input", {
            method: "POST",
            headers: { "Authorization": `Bearer ${token}` },
            body: formData,
        });

        const data = await resp.json();

        if (!resp.ok) {
            appendMessage("agent", `🎤 Transcription failed: ${data.detail || "Unknown error"}. Please type your symptoms.`);
            return;
        }

        const transcript = data.transcript;
        // Show transcript to user for review
        appendMessage("agent", `🎤 I heard: <b>"${transcript}"</b><br/>Sending to healthcare agent…`);

        // Auto-populate input and submit
        const inputEl = document.getElementById("chat-input");
        if (inputEl) {
            inputEl.value = transcript;
        }
        // Small delay so user can see the transcript, then send
        setTimeout(() => sendMessage(), 600);

    } catch (err) {
        appendMessage("agent", "🎤 Could not reach transcription service. Please type your symptoms.");
        console.error("Voice upload error:", err);
    }
}
