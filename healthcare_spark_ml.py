from __future__ import annotations

import os
from typing import Any


RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
    3: "Emergency",
}

FEATURE_COLUMNS = [
    "age",
    "chest_pain",
    "diabetes",
    "hypertension",
    "shortness_of_breath",
    "blood_pressure",
]


def _spark_session(app_name: str):
    from pyspark.sql import SparkSession

    spark_master = os.getenv("SPARK_MASTER_URL", "local[*]")
    return SparkSession.builder.appName(app_name).master(spark_master).getOrCreate()


def _hdfs_base_path() -> str:
    return os.getenv("HDFS_ANALYTICS_PATH", "hdfs://namenode:9000/healthcare").rstrip("/")


def _risk_training_path() -> str:
    return os.getenv("HDFS_RISK_TRAINING_PATH", f"{_hdfs_base_path()}/patients.csv")


def _risk_model_path() -> str:
    return os.getenv("HDFS_RISK_MODEL_PATH", f"{_hdfs_base_path()}/models/risk_random_forest")


def _coerce_patient_features(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "age": int(payload.get("age", 55)),
        "chest_pain": int(bool(payload.get("chest_pain", 0))),
        "diabetes": int(bool(payload.get("diabetes", 0))),
        "hypertension": int(bool(payload.get("hypertension", 0))),
        "shortness_of_breath": int(bool(payload.get("shortness_of_breath", 0))),
        "blood_pressure": int(payload.get("blood_pressure", 120)),
    }


def train_risk_model_from_hdfs() -> dict[str, Any]:
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql.functions import col

    training_path = _risk_training_path()
    model_path = _risk_model_path()
    spark = _spark_session("HealthcareRiskMLTraining")

    try:
        df = spark.read.csv(training_path, header=True, inferSchema=True)
        df = df.select(
            *(col(name).cast("double").alias(name) for name in FEATURE_COLUMNS),
            col("risk").cast("double").alias("risk"),
        ).dropna()

        total_rows = df.count()
        if total_rows == 0:
            raise ValueError(f"No training rows found at {training_path}")

        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        if train_df.count() == 0:
            train_df = df
        if test_df.count() == 0:
            test_df = df

        assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features")
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="risk",
            predictionCol="prediction",
            probabilityCol="probability",
            numTrees=50,
            maxDepth=5,
            seed=42,
        )

        pipeline = Pipeline(stages=[assembler, rf])
        model = pipeline.fit(train_df)
        model.write().overwrite().save(model_path)

        predictions = model.transform(test_df)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="risk",
            predictionCol="prediction",
            metricName="accuracy",
        )
        accuracy = evaluator.evaluate(predictions)
        rf_model = model.stages[-1]

        return {
            "status": "trained",
            "training_path": training_path,
            "model_path": model_path,
            "rows": total_rows,
            "accuracy": round(float(accuracy), 3),
            "risk_labels": RISK_LABELS,
            "feature_importance": [
                {"feature": feature, "importance": round(float(score), 4)}
                for feature, score in zip(FEATURE_COLUMNS, rf_model.featureImportances)
            ],
        }
    finally:
        spark.stop()


def predict_risk_with_spark_ml(payload: dict[str, Any]) -> dict[str, Any]:
    from pyspark.ml.pipeline import PipelineModel

    features = _coerce_patient_features(payload)
    model_path = _risk_model_path()
    spark = _spark_session("HealthcareRiskMLPrediction")

    try:
        model = PipelineModel.load(model_path)
        df = spark.createDataFrame([features])
        row = model.transform(df).select("prediction", "probability").first()

        prediction_index = int(row["prediction"])
        probabilities = [float(value) for value in row["probability"].toArray()]
        probability_map = {
            RISK_LABELS[index]: round(probability, 3)
            for index, probability in enumerate(probabilities)
            if index in RISK_LABELS
        }
        weighted_score = sum(index * probability for index, probability in enumerate(probabilities))
        risk_score = round((weighted_score / (len(RISK_LABELS) - 1)) * 100, 1)

        return {
            "source": "pyspark_ml",
            "model_path": model_path,
            "prediction": RISK_LABELS.get(prediction_index, "Unknown"),
            "risk_score": risk_score,
            "probabilities": probability_map,
            "features": features,
        }
    finally:
        spark.stop()
