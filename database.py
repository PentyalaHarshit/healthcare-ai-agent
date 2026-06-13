import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, text
from sqlalchemy.orm import declarative_base, sessionmaker
from passlib.context import CryptContext

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:healthcare123@localhost:5432/healthcare")

# Fallback to SQLite if PostgreSQL fails to connect
try:
    if DATABASE_URL.startswith("postgresql"):
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 5})
        conn = engine.connect()
        conn.close()
        logger_info = "✅ Database: PostgreSQL connected"
    else:
        raise ValueError("SQLite fallback")
except Exception:
    DATABASE_URL = "sqlite:///./healthcare.db"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
    logger_info = "⚠️ Database: Fallback to SQLite"

print(logger_info.encode("ascii", "ignore").decode("ascii").strip())

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String)  # admin, doctor, patient


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    hospital = Column(String, default="Texas Health Frisco")
    specialty = Column(String)
    experience = Column(Integer, default=5)
    available_time = Column(String)
    available = Column(Boolean, default=True)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer, nullable=True)
    location = Column(String, nullable=True)
    medical_history = Column(String, nullable=True)
    created_at = Column(String, nullable=True)


class Slot(Base):
    __tablename__ = "slots"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer)
    date = Column(String)
    time = Column(String)
    available = Column(Integer, default=1)  # 1 for True, 0 for False


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=True)
    doctor_id = Column(Integer, nullable=True)
    patient_name = Column(String)
    doctor_name = Column(String)
    hospital = Column(String, nullable=True)
    specialty = Column(String)
    appointment_date = Column(String)
    appointment_time = Column(String)
    symptoms = Column(String, nullable=True)
    urgency = Column(String)
    status = Column(String, default="Confirmed")
    calendar_event_id = Column(String, nullable=True)
    created_at = Column(String, nullable=True)


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=True)
    patient_name = Column(String)
    hospital = Column(String, nullable=True)
    symptoms = Column(String)
    disease = Column(String, nullable=True)
    specialty = Column(String, nullable=True)
    risk_score = Column(Integer)
    priority = Column(String)
    recommendation = Column(String)
    details = Column(String, nullable=True)
    generated_at = Column(String, nullable=True)


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=True)
    session_id = Column(String, index=True)
    message = Column(String)
    reply = Column(String)
    timestamp = Column(String)


class HealthRule(Base):
    __tablename__ = "health_rules"

    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String)
    category = Column(String)
    specialty = Column(String)
    priority = Column(String)
    risk_points = Column(Integer)
    multiplier = Column(Float)
    explanation = Column(String)


def init_db():
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            doctor_columns = {
                row[1] for row in conn.execute(text("PRAGMA table_info(doctors)")).fetchall()
            }
            doctor_migrations = {
                "hospital": "ALTER TABLE doctors ADD COLUMN hospital VARCHAR DEFAULT 'Texas Health Frisco'",
                "experience": "ALTER TABLE doctors ADD COLUMN experience INTEGER DEFAULT 5",
                "available_time": "ALTER TABLE doctors ADD COLUMN available_time VARCHAR",
                "available": "ALTER TABLE doctors ADD COLUMN available BOOLEAN DEFAULT 1",
            }
            for column, statement in doctor_migrations.items():
                if column not in doctor_columns:
                    conn.execute(text(statement))

            table_migrations = {
                "appointments": {
                    "patient_id": "ALTER TABLE appointments ADD COLUMN patient_id INTEGER",
                    "doctor_id": "ALTER TABLE appointments ADD COLUMN doctor_id INTEGER",
                    "hospital": "ALTER TABLE appointments ADD COLUMN hospital VARCHAR",
                    "symptoms": "ALTER TABLE appointments ADD COLUMN symptoms VARCHAR",
                    "created_at": "ALTER TABLE appointments ADD COLUMN created_at VARCHAR",
                },
                "reports": {
                    "patient_id": "ALTER TABLE reports ADD COLUMN patient_id INTEGER",
                    "hospital": "ALTER TABLE reports ADD COLUMN hospital VARCHAR",
                    "disease": "ALTER TABLE reports ADD COLUMN disease VARCHAR",
                    "specialty": "ALTER TABLE reports ADD COLUMN specialty VARCHAR",
                    "generated_at": "ALTER TABLE reports ADD COLUMN generated_at VARCHAR",
                },
                "chat_history": {
                    "patient_id": "ALTER TABLE chat_history ADD COLUMN patient_id INTEGER",
                },
            }
            for table_name, migrations in table_migrations.items():
                existing_columns = {
                    row[1] for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                }
                for column, statement in migrations.items():
                    if column not in existing_columns:
                        conn.execute(text(statement))

            patient_tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
            ).fetchall()
            if not patient_tables:
                conn.execute(text(
                    "CREATE TABLE patients ("
                    "id INTEGER PRIMARY KEY, "
                    "name VARCHAR, "
                    "age INTEGER, "
                    "location VARCHAR, "
                    "medical_history VARCHAR, "
                    "created_at VARCHAR)"
                ))

            patient_columns = {
                row[1] for row in conn.execute(text("PRAGMA table_info(patients)")).fetchall()
            }
            patient_migrations = {
                "age": "ALTER TABLE patients ADD COLUMN age INTEGER",
                "location": "ALTER TABLE patients ADD COLUMN location VARCHAR",
                "medical_history": "ALTER TABLE patients ADD COLUMN medical_history VARCHAR",
                "created_at": "ALTER TABLE patients ADD COLUMN created_at VARCHAR",
            }
            for column, statement in patient_migrations.items():
                if column not in patient_columns:
                    conn.execute(text(statement))

    db = SessionLocal()

    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

    # Seed users
    if db.query(User).count() == 0:
        db.add_all([
            User(username="admin", password_hash=pwd_context.hash("admin123"), role="admin"),
            User(username="doctor", password_hash=pwd_context.hash("doctor123"), role="doctor"),
            User(username="patient", password_hash=pwd_context.hash("patient123"), role="patient"),
        ])
        db.commit()

    # Seed / refresh sample doctors
    seed_doctors = [
        Doctor(id=1, name="Dr. Smith", hospital="Texas Health Frisco", specialty="Cardiology", experience=15, available_time="10:00 AM"),
        Doctor(id=2, name="Dr. Johnson", hospital="Texas Health Frisco", specialty="Cardiology", experience=12, available_time="11:30 AM"),
        Doctor(id=3, name="Dr. Lee", hospital="Texas Health Frisco", specialty="Pulmonology", experience=14, available_time="11:30 AM"),
        Doctor(id=4, name="Dr. Brown", hospital="Texas Health Frisco", specialty="General Physician", experience=20, available_time="02:00 PM"),
        Doctor(id=5, name="Dr. Patel", hospital="Texas Health Frisco", specialty="Cardiology", experience=10, available_time="2:00 PM"),
        Doctor(id=6, name="Dr. Nguyen", hospital="Texas Health Frisco", specialty="Neurology", experience=8, available_time="01:00 PM"),
        Doctor(id=7, name="Dr. Okafor", hospital="Baylor Scott & White Plano", specialty="Endocrinology", experience=11, available_time="10:30 AM"),
        Doctor(id=8, name="Dr. Martinez", hospital="Baylor Scott & White Plano", specialty="Cardiology", experience=9, available_time="03:00 PM"),
        Doctor(id=9, name="Dr. Chen", hospital="Medical City Plano", specialty="Cardiology", experience=13, available_time="09:30 AM"),
        Doctor(id=10, name="Dr. Wilson", hospital="Texas Health Plano", specialty="General Physician", experience=16, available_time="02:30 PM"),
        Doctor(id=11, name="Dr. Adams", hospital="Children's Medical Center Plano", specialty="Pediatrics", experience=7, available_time="10:00 AM"),
    ]
    for doctor in seed_doctors:
        existing = db.get(Doctor, doctor.id)
        if existing:
            existing.name = doctor.name
            existing.hospital = doctor.hospital
            existing.specialty = doctor.specialty
            existing.experience = doctor.experience
            existing.available_time = doctor.available_time
            existing.available = True
        else:
            db.add(doctor)
    db.commit()

    # Seed / refresh sample slots
    seed_slots = [
        Slot(id=1, doctor_id=1, date="2026-06-20", time="10:00 AM", available=1),
        Slot(id=2, doctor_id=2, date="2026-06-20", time="11:30 AM", available=1),
        Slot(id=3, doctor_id=3, date="2026-06-20", time="11:30 AM", available=1),
        Slot(id=4, doctor_id=4, date="2026-06-22", time="02:00 PM", available=1),
        Slot(id=5, doctor_id=5, date="2026-06-20", time="2:00 PM", available=1),
        Slot(id=6, doctor_id=6, date="2026-06-21", time="01:00 PM", available=1),
        Slot(id=7, doctor_id=7, date="2026-06-22", time="10:30 AM", available=1),
        Slot(id=8, doctor_id=8, date="2026-06-20", time="03:00 PM", available=1),
        Slot(id=9, doctor_id=9, date="2026-06-20", time="09:30 AM", available=1),
        Slot(id=10, doctor_id=10, date="2026-06-22", time="02:30 PM", available=1),
        Slot(id=11, doctor_id=11, date="2026-06-23", time="10:00 AM", available=1),
    ]
    for slot in seed_slots:
        existing = db.get(Slot, slot.id)
        if existing:
            existing.doctor_id = slot.doctor_id
            existing.date = slot.date
            existing.time = slot.time
            existing.available = slot.available
        else:
            db.add(slot)
    db.commit()

    # Seed health rules
    if db.query(HealthRule).count() == 0:
        db.add_all([
            HealthRule(
                keyword="chest pain",
                category="symptom",
                specialty="Cardiology",
                priority="High",
                risk_points=40,
                multiplier=3.0,
                explanation="Chest pain detected, which may require heart-related evaluation."
            ),
            HealthRule(
                keyword="shortness of breath",
                category="symptom",
                specialty="Cardiology",
                priority="High",
                risk_points=30,
                multiplier=2.5,
                explanation="Shortness of breath detected, which increases urgency."
            ),
            HealthRule(
                keyword="breathing",
                category="symptom",
                specialty="Pulmonology",
                priority="High",
                risk_points=30,
                multiplier=2.0,
                explanation="Breathing difficulty detected."
            ),
            HealthRule(
                keyword="diabetes",
                category="history",
                specialty="Endocrinology",
                priority="Medium",
                risk_points=20,
                multiplier=1.8,
                explanation="Diabetes history detected."
            ),
            HealthRule(
                keyword="high blood pressure",
                category="history",
                specialty="Cardiology",
                priority="Medium",
                risk_points=15,
                multiplier=1.6,
                explanation="High blood pressure can increase heart risk."
            ),
            HealthRule(
                keyword="hypertension",
                category="history",
                specialty="Cardiology",
                priority="Medium",
                risk_points=15,
                multiplier=1.6,
                explanation="Hypertension can increase heart risk."
            ),
            HealthRule(
                keyword="headache",
                category="symptom",
                specialty="Neurology",
                priority="Medium",
                risk_points=20,
                multiplier=1.3,
                explanation="Headache detected. Neurology consultation may be useful."
            ),
            HealthRule(
                keyword="rash",
                category="symptom",
                specialty="Dermatology",
                priority="Normal",
                risk_points=10,
                multiplier=1.1,
                explanation="Rash detected. Dermatology consultation may be useful."
            ),
            HealthRule(
                keyword="stomach pain",
                category="symptom",
                specialty="Gastroenterology",
                priority="Medium",
                risk_points=20,
                multiplier=1.3,
                explanation="Stomach pain detected. Gastroenterology consultation may be useful."
            ),
        ])
        db.commit()

    db.close()
