from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine(
    "sqlite:///./healthcare.db",
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    specialty = Column(String)
    experience = Column(Integer)
    available_time = Column(String)
    available = Column(Boolean, default=True)


class HealthRule(Base):
    __tablename__ = "health_rules"

    id = Column(Integer, primary_key=True)
    keyword = Column(String)
    category = Column(String)
    specialty = Column(String)
    priority = Column(String)
    risk_points = Column(Integer)
    multiplier = Column(Float)
    explanation = Column(String)


def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    if db.query(Doctor).count() == 0:
        db.add_all([
            Doctor(name="Dr. John Smith", specialty="Cardiology", experience=15, available_time="10:00 AM"),
            Doctor(name="Dr. Sarah Johnson", specialty="Cardiology", experience=12, available_time="2:00 PM"),
            Doctor(name="Dr. Michael Brown", specialty="Endocrinology", experience=10, available_time="11:00 AM"),
            Doctor(name="Dr. Emily Davis", specialty="Neurology", experience=8, available_time="1:00 PM"),
            Doctor(name="Dr. Jennifer Lee", specialty="Dermatology", experience=7, available_time="9:30 AM"),
            Doctor(name="Dr. David Miller", specialty="General Physician", experience=20, available_time="4:00 PM"),
            Doctor(name="Dr. James Taylor", specialty="Pulmonology", experience=14, available_time="10:30 AM"),
            Doctor(name="Dr. Maria Garcia", specialty="Gastroenterology", experience=11, available_time="1:30 PM"),
        ])
        db.commit()

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