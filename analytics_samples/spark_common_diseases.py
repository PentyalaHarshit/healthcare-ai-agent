from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("HealthcareAnalytics").getOrCreate()

df = spark.read.csv(
    "hdfs://namenode:9000/healthcare/appointments.csv",
    header=True,
    inferSchema=True,
)

df.groupBy("symptoms").count().orderBy("count", ascending=False).show()

spark.stop()
