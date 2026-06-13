from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


FEATURE_COLUMNS = [
    "age",
    "chest_pain",
    "diabetes",
    "hypertension",
    "shortness_of_breath",
    "blood_pressure",
]

TRAINING_PATH = "hdfs://namenode:9000/healthcare/patients.csv"
MODEL_PATH = "hdfs://namenode:9000/healthcare/models/risk_random_forest"


spark = SparkSession.builder.appName("HealthcareRiskMLTraining").getOrCreate()

df = spark.read.csv(TRAINING_PATH, header=True, inferSchema=True).select(
    *(col(name).cast("double").alias(name) for name in FEATURE_COLUMNS),
    col("risk").cast("double").alias("risk"),
).dropna()

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

model = Pipeline(stages=[assembler, rf]).fit(train_df)
model.write().overwrite().save(MODEL_PATH)

predictions = model.transform(test_df)
accuracy = MulticlassClassificationEvaluator(
    labelCol="risk",
    predictionCol="prediction",
    metricName="accuracy",
).evaluate(predictions)

print(f"Rows: {df.count()}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Model saved: {MODEL_PATH}")

predictions.select(
    "age",
    "chest_pain",
    "diabetes",
    "hypertension",
    "shortness_of_breath",
    "blood_pressure",
    "risk",
    "prediction",
    "probability",
).show(truncate=False)

spark.stop()
