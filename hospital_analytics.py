from __future__ import annotations

import os
from collections import Counter
from typing import Any

import pandas as pd
from sqlalchemy import func

from database import Appointment, Doctor, Patient, Report, SessionLocal


def _rows_to_dict(rows) -> list[dict[str, Any]]:
    return [dict(row._mapping) for row in rows]


def get_hospital_analytics() -> dict[str, Any]:
    db = SessionLocal()
    try:
        disease_rows = db.query(
            Report.disease.label("disease"),
            func.count(Report.id).label("count"),
        ).filter(
            Report.disease.isnot(None)
        ).group_by(
            Report.disease
        ).order_by(
            func.count(Report.id).desc()
        ).limit(10).all()

        doctor_rows = db.query(
            Appointment.doctor_name.label("doctor"),
            Appointment.specialty.label("specialty"),
            func.count(Appointment.id).label("appointments"),
        ).filter(
            Appointment.doctor_name.isnot(None)
        ).group_by(
            Appointment.doctor_name,
            Appointment.specialty,
        ).order_by(
            func.count(Appointment.id).desc()
        ).limit(10).all()

        load_rows = db.query(
            Appointment.hospital.label("hospital"),
            func.count(Appointment.id).label("appointments"),
        ).filter(
            Appointment.hospital.isnot(None)
        ).group_by(
            Appointment.hospital
        ).order_by(
            func.count(Appointment.id).desc()
        ).all()

        emergency_rows = db.query(
            func.coalesce(Report.generated_at, "").label("generated_at"),
            Report.priority.label("priority"),
        ).filter(
            Report.priority.in_(["Emergency", "High"])
        ).all()

        emergency_counter = Counter()
        for generated_at, priority in emergency_rows:
            day = (generated_at or "unknown")[:10] or "unknown"
            emergency_counter[(day, priority)] += 1

        emergency_trends = [
            {"date": day, "priority": priority, "count": count}
            for (day, priority), count in sorted(emergency_counter.items())
        ]

        totals = {
            "patients": db.query(func.count(Patient.id)).scalar() or 0,
            "doctors": db.query(func.count(Doctor.id)).scalar() or 0,
            "appointments": db.query(func.count(Appointment.id)).scalar() or 0,
            "reports": db.query(func.count(Report.id)).scalar() or 0,
            "emergency_reports": db.query(func.count(Report.id)).filter(
                Report.priority == "Emergency"
            ).scalar() or 0,
        }

        return {
            "source": "postgresql",
            "questions": {
                "most_common_disease": _rows_to_dict(disease_rows),
                "most_busy_doctor": _rows_to_dict(doctor_rows),
                "hospital_load": _rows_to_dict(load_rows),
                "emergency_trends": emergency_trends,
            },
            "totals": totals,
        }
    finally:
        db.close()


def export_postgres_to_hdfs() -> dict[str, Any]:
    """Export analytics source tables to HDFS as Parquet using Spark.

    This is intended for the Hadoop path:
    PostgreSQL -> Spark DataFrames -> HDFS parquet -> Spark analytics jobs.
    """
    from pyspark.sql import SparkSession

    hdfs_path = os.getenv("HDFS_ANALYTICS_PATH", "hdfs://namenode:9000/healthcare")
    spark_master = os.getenv("SPARK_MASTER_URL", "local[*]")

    db = SessionLocal()
    try:
        reports_df = pd.read_sql(db.query(Report).statement, db.bind)
        appointments_df = pd.read_sql(db.query(Appointment).statement, db.bind)
        doctors_df = pd.read_sql(db.query(Doctor).statement, db.bind)
    finally:
        db.close()

    spark = SparkSession.builder.appName("HospitalAnalyticsExport").master(spark_master).getOrCreate()
    try:
        spark.createDataFrame(reports_df).write.mode("overwrite").parquet(f"{hdfs_path}/reports")
        spark.createDataFrame(appointments_df).write.mode("overwrite").parquet(f"{hdfs_path}/appointments")
        spark.createDataFrame(doctors_df).write.mode("overwrite").parquet(f"{hdfs_path}/doctors")
        return {
            "status": "success",
            "hdfs_path": hdfs_path,
            "tables": ["reports", "appointments", "doctors"],
        }
    finally:
        spark.stop()


def run_spark_hdfs_analytics() -> dict[str, Any]:
    from pyspark.sql import SparkSession

    hdfs_path = os.getenv("HDFS_ANALYTICS_PATH", "hdfs://namenode:9000/healthcare")
    spark_master = os.getenv("SPARK_MASTER_URL", "local[*]")
    spark = SparkSession.builder.appName("HospitalAnalytics").master(spark_master).getOrCreate()
    try:
        reports = spark.read.parquet(f"{hdfs_path}/reports")
        appointments = spark.read.parquet(f"{hdfs_path}/appointments")

        disease = [
            row.asDict()
            for row in reports.groupBy("disease").count().orderBy("count", ascending=False).limit(10).collect()
        ]
        busy_doctor = [
            row.asDict()
            for row in appointments.groupBy("doctor_name", "specialty").count().orderBy("count", ascending=False).limit(10).collect()
        ]
        load = [
            row.asDict()
            for row in appointments.groupBy("hospital").count().orderBy("count", ascending=False).collect()
        ]
        emergency = reports.filter(
            reports.priority.isin("Emergency", "High")
        ).groupBy(
            reports.generated_at.substr(1, 10).alias("date"),
            "priority",
        ).count().orderBy("date").collect()

        return {
            "source": "hdfs_spark",
            "questions": {
                "most_common_disease": disease,
                "most_busy_doctor": busy_doctor,
                "hospital_load": load,
                "emergency_trends": [row.asDict() for row in emergency],
            },
        }
    finally:
        spark.stop()


def _hdfs_appointments_csv_path() -> str:
    base_path = os.getenv("HDFS_ANALYTICS_PATH", "hdfs://namenode:9000/healthcare").rstrip("/")
    return f"{base_path}/appointments.csv"


def _spark_session(app_name: str):
    from pyspark.sql import SparkSession

    spark_master = os.getenv("SPARK_MASTER_URL", "local[*]")
    return SparkSession.builder.appName(app_name).master(spark_master).getOrCreate()


def read_hdfs_appointments_csv():
    spark = _spark_session("HealthcareCsvAnalytics")
    try:
        df = spark.read.csv(_hdfs_appointments_csv_path(), header=True, inferSchema=True)
        return spark, df
    except Exception:
        spark.stop()
        raise


def common_diseases_from_hdfs(limit: int = 10) -> list[dict[str, Any]]:
    spark, df = read_hdfs_appointments_csv()
    try:
        return [
            row.asDict()
            for row in df.groupBy("symptoms")
            .count()
            .orderBy("count", ascending=False)
            .limit(limit)
            .collect()
        ]
    finally:
        spark.stop()


def busy_doctors_from_hdfs(limit: int = 10) -> list[dict[str, Any]]:
    spark, df = read_hdfs_appointments_csv()
    try:
        if "doctor" not in df.columns:
            return []
        return [
            row.asDict()
            for row in df.groupBy("doctor")
            .count()
            .orderBy("count", ascending=False)
            .limit(limit)
            .collect()
        ]
    finally:
        spark.stop()


def hospital_load_from_hdfs() -> list[dict[str, Any]]:
    spark, df = read_hdfs_appointments_csv()
    try:
        return [
            row.asDict()
            for row in df.groupBy("hospital")
            .count()
            .orderBy("count", ascending=False)
            .collect()
        ]
    finally:
        spark.stop()


def emergency_trends_from_hdfs() -> list[dict[str, Any]]:
    spark, df = read_hdfs_appointments_csv()
    try:
        if "priority" not in df.columns:
            return []
        return [
            row.asDict()
            for row in df.filter(df.priority == "Emergency")
            .groupBy("priority")
            .count()
            .collect()
        ]
    finally:
        spark.stop()
