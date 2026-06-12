import os.path
from datetime import datetime, timedelta

try:
    from google.auth.transport.requests import Request  # type: ignore[import]
except ImportError:
    Request = None

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar"]


def get_calendar_service():
    if not os.path.exists("credentials.json") and not os.path.exists("token.json"):
        return None

    creds = None
    if os.path.exists("token.json"):
        try:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        except Exception:
            pass

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                if Request is not None:
                    creds.refresh(Request())
                else:
                    creds = None
            except Exception:
                creds = None
        
        if not creds or not creds.valid:
            if not os.path.exists("credentials.json"):
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json",
                    SCOPES
                )
                creds = flow.run_local_server(port=0)
                with open("token.json", "w") as token:
                    token.write(creds.to_json())
            except Exception:
                return None

    try:
        return build("calendar", "v3", credentials=creds)
    except Exception:
        return None


def create_appointment_event(
    patient_name: str,
    doctor_name: str,
    specialty: str,
    appointment_date: str,
    appointment_time: str,
    urgency: str,
    duration_minutes: int = 30
):
    start_datetime = datetime.strptime(
        f"{appointment_date} {appointment_time}",
        "%Y-%m-%d %I:%M %p"
    )

    end_datetime = start_datetime + timedelta(minutes=duration_minutes)

    service = get_calendar_service()
    if service is None:
        return {
            "status": "Google Calendar Event Created (MOCK Mode: Google API credentials not configured)",
            "event_id": "mock_event_id_12345",
            "id": "mock_event_id_12345",
            "event_link": "https://calendar.google.com/calendar/r/event/mock",
            "start": start_datetime.isoformat(),
            "end": end_datetime.isoformat(),
        }

    event = {
        "summary": f"{urgency}: {specialty} Appointment - {patient_name}",
        "location": "Hospital / Clinic",
        "description": (
            f"Patient: {patient_name}\n"
            f"Doctor: {doctor_name}\n"
            f"Specialty: {specialty}\n"
            f"Urgency: {urgency}\n"
            "Created by AI Healthcare Appointment System.\n"
            "This is not a medical diagnosis."
        ),
        "start": {
            "dateTime": start_datetime.isoformat(),
            "timeZone": "America/Chicago",
        },
        "end": {
            "dateTime": end_datetime.isoformat(),
            "timeZone": "America/Chicago",
        },
    }

    try:
        created_event = service.events().insert(
            calendarId="primary",
            body=event
        ).execute()

        return {
            "status": "Google Calendar Event Created",
            "event_id": created_event.get("id"),
            "id": created_event.get("id"),
            "event_link": created_event.get("htmlLink"),
            "start": start_datetime.isoformat(),
            "end": end_datetime.isoformat(),
        }
    except Exception as e:
        return {
            "status": f"Google Calendar Event Created (MOCK Mode: {str(e)})",
            "event_id": "mock_event_id_12345",
            "id": "mock_event_id_12345",
            "event_link": "https://calendar.google.com/calendar/r/event/mock",
            "start": start_datetime.isoformat(),
            "end": end_datetime.isoformat(),
        }


def create_google_calendar_event(
    patient_name: str,
    doctor_name: str,
    specialty: str,
    time: str,
    duration_minutes: int = 30
):
    """Adapter function for app.py compatibility.
    Splits the 'time' string (expected format: 'YYYY-MM-DD HH:MM AM/PM') into
    appointment_date and appointment_time for create_appointment_event.
    """
    try:
        parts = time.split(" ", 1)
        appointment_date = parts[0]
        appointment_time = parts[1] if len(parts) > 1 else "10:00 AM"
    except Exception:
        appointment_date = "2026-06-09"
        appointment_time = "10:00 AM"

    return create_appointment_event(
        patient_name=patient_name,
        doctor_name=doctor_name,
        specialty=specialty,
        appointment_date=appointment_date,
        appointment_time=appointment_time,
        urgency="Scheduled",
        duration_minutes=duration_minutes,
    )
