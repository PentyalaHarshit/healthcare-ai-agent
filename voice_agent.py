"""
Voice Agent — Upgrade #5
Speech-to-Text:  OpenAI Whisper (local model)
Text-to-Speech:  ElevenLabs API (optional, falls back to pyttsx3 then skip)
Flow: audio bytes → Whisper STT → healthcare agent → (optional) TTS response
"""
import io
import os
import logging
import tempfile

logger = logging.getLogger(__name__)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")  # tiny | base | small | medium

# ── Load Whisper once at startup ──────────────────────────────────────────────
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info(f"✅ Whisper '{WHISPER_MODEL_SIZE}' model loaded")
        return _whisper_model
    except Exception as e:
        logger.warning(f"⚠️  Whisper not available: {e}")
        return None


# ── STT: audio bytes → transcript ────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes, content_type: str = "audio/webm") -> dict:
    """
    Transcribe audio bytes using Whisper.
    Returns {"transcript": str, "language": str, "method": str}
    """
    model = _get_whisper()

    if model is None:
        # Last-resort: return placeholder so UI can handle gracefully
        return {
            "transcript": "",
            "language": "en",
            "method": "unavailable",
            "error": "Whisper not installed. Run: pip install openai-whisper",
        }

    # Write bytes to a temp file (Whisper needs a file path)
    ext = "webm"
    if "ogg" in content_type:
        ext = "ogg"
    elif "wav" in content_type:
        ext = "wav"
    elif "mp3" in content_type:
        ext = "mp3"

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, fp16=False)
        transcript = result.get("text", "").strip()
        language = result.get("language", "en")
        return {"transcript": transcript, "language": language, "method": "whisper"}
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return {"transcript": "", "language": "en", "method": "whisper_error", "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── TTS: text → audio bytes ───────────────────────────────────────────────────
def synthesize_speech(text: str, voice_id: str = "rachel") -> bytes | None:
    """
    Convert text to speech.
    Priority: ElevenLabs API → pyttsx3 → None (frontend uses browser TTS)
    Returns audio bytes or None.
    """
    if ELEVENLABS_API_KEY:
        try:
            return _elevenlabs_tts(text, voice_id)
        except Exception as e:
            logger.warning(f"ElevenLabs TTS failed: {e}")

    # Fallback: pyttsx3 (offline)
    try:
        return _pyttsx3_tts(text)
    except Exception as e:
        logger.warning(f"pyttsx3 TTS failed: {e}")

    return None


def _elevenlabs_tts(text: str, voice_id: str) -> bytes:
    import requests

    VOICE_IDS = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "adam":   "pNInz6obpgDQGcFmaJgB",
        "bella":  "EXAVITQu4vr4xnSDxMaL",
    }
    vid = VOICE_IDS.get(voice_id.lower(), VOICE_IDS["rachel"])

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "text": text[:500],
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.content


def _pyttsx3_tts(text: str) -> bytes:
    import pyttsx3
    import wave

    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        engine.save_to_file(text[:500], tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
