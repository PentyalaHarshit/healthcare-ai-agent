from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SparkSession


RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
    3: "Emergency",
}

MODEL_PATH = "hdfs://namenode:9000/healthcare/models/risk_random_forest"


spark = SparkSession.builder.appName("HealthcareRiskMLPrediction").getOrCreate()

model = PipelineModel.load(MODEL_PATH)
patient = {
    "age": 64,
    "chest_pain": 1,
    "diabetes": 1,
    "hypertension": 1,
    "shortness_of_breath": 1,
    "blood_pressure": 158,
}

row = model.transform(spark.createDataFrame([patient])).select(
    "prediction",
    "probability",
).first()

prediction = int(row["prediction"])
probabilities = [float(value) for value in row["probability"].toArray()]

print(f"Prediction: {RISK_LABELS[prediction]}")
print(
    "Probabilities: "
    + ", ".join(
        f"{RISK_LABELS[index]}={probability:.3f}"
        for index, probability in enumerate(probabilities)
    )
)

spark.stop()
