# Cloud Deployment

Production target:

```text
Browser
  |
  v
Nginx HTTPS reverse proxy
  |
  v
FastAPI main app
  |
  v
CrewAI-style healthcare agent pipeline
  |
  v
RAG / Chroma vector store
  |
  v
MCP server
  |
  v
PostgreSQL
```

Analytics pipeline for large hospital volume:

```text
Millions of appointments
  |
  v
PostgreSQL operational store
  |
  v
Spark export job
  |
  v
HDFS parquet lake
  |
  v
Spark analytics
  |
  v
Hospital Analytics Dashboard
```

## AWS EC2

Recommended instance for the current ML/RAG dependencies:

- Ubuntu 22.04 or 24.04 LTS
- `t3.large` minimum, `t3.xlarge` preferred for smoother model/RAG startup
- 30 GB or more gp3 EBS volume
- Security group inbound rules:
  - `22/tcp` from your IP only
  - `80/tcp` from `0.0.0.0/0`
  - `443/tcp` from `0.0.0.0/0`

Point your domain DNS `A` record to the EC2 public IPv4 address before requesting HTTPS certificates.

## Install Docker

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ubuntu
newgrp docker
```

## Configure Environment

```bash
git clone <your-repo-url> health_ai_project
cd health_ai_project
cp .env.production.example .env.production
nano .env.production
```

Set:

- `DOMAIN_NAME` to your real domain
- `POSTGRES_PASSWORD` to a strong password
- optional voice keys if you use voice features

## First HTTPS Certificate

Start Nginx in HTTP-only mode so Certbot can complete the ACME challenge:

```bash
mkdir -p deploy/certbot/www deploy/certbot/conf
docker compose --env-file .env.production -f docker-compose.prod.yml -f docker-compose.bootstrap.yml up -d --build postgres mcp_server main_app nginx
```

Request the certificate:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml --profile certbot run --rm certbot certonly \
  --webroot \
  --webroot-path /var/www/certbot \
  --email your-email@example.com \
  --agree-tos \
  --no-eff-email \
  -d your-domain.example.com
```

Replace `your-email@example.com` and `your-domain.example.com`.

## Start Production Stack

After the certificate exists, restart with the HTTPS Nginx config:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml down
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

Open:

```text
https://your-domain.example.com
```

## Certificate Renewal

Add a cron job:

```bash
crontab -e
```

```cron
0 3 * * * cd /home/ubuntu/health_ai_project && docker compose --env-file .env.production -f docker-compose.prod.yml --profile certbot run --rm certbot renew && docker compose --env-file .env.production -f docker-compose.prod.yml exec -T nginx nginx -s reload
```

## Operations

View logs:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml logs -f main_app
docker compose --env-file .env.production -f docker-compose.prod.yml logs -f mcp_server
docker compose --env-file .env.production -f docker-compose.prod.yml logs -f nginx
```

Restart after code changes:

```bash
git pull
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

Back up PostgreSQL:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml exec postgres pg_dump -U healthcare healthcare > healthcare_backup.sql
```

Restore PostgreSQL:

```bash
cat healthcare_backup.sql | docker compose --env-file .env.production -f docker-compose.prod.yml exec -T postgres psql -U healthcare healthcare
```

## Hadoop + Spark Analytics

The production compose file includes optional Hadoop and Spark services under the `analytics` profile.

Start the full production stack with analytics:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml --profile analytics up -d --build
```

Open dashboards:

```text
Hospital Analytics: https://your-domain.example.com/hospital-analytics
HDFS NameNode: http://your-ec2-public-ip:9870
Spark Master: http://your-ec2-public-ip:18080
```

For stricter production networking, restrict ports `9870` and `18080` to your IP in the EC2 security group, or remove those public port mappings.

Export PostgreSQL operational data to HDFS:

```bash
curl -X POST https://your-domain.example.com/api/hospital-analytics/export-hdfs
```

Upload a CSV directly to HDFS:

```bash
docker cp analytics_samples/appointments.csv healthcare_hdfs_namenode:/tmp/appointments.csv
docker compose --env-file .env.production -f docker-compose.prod.yml --profile analytics exec -T namenode hdfs dfs -mkdir -p /healthcare
docker compose --env-file .env.production -f docker-compose.prod.yml --profile analytics exec -T namenode hdfs dfs -put -f /tmp/appointments.csv /healthcare/appointments.csv
docker compose --env-file .env.production -f docker-compose.prod.yml --profile analytics exec -T namenode hdfs dfs -ls /healthcare
```

Run Spark analytics from HDFS:

```bash
curl https://your-domain.example.com/api/hospital-analytics/spark
```

CSV-specific Spark endpoints:

```bash
curl https://your-domain.example.com/analytics/common-diseases
curl https://your-domain.example.com/analytics/busy-doctors
curl https://your-domain.example.com/analytics/hospital-load
curl https://your-domain.example.com/analytics/emergency-trends
```

The normal dashboard endpoint reads PostgreSQL directly:

```bash
curl https://your-domain.example.com/api/hospital-analytics
```

This answers:

- Most common disease
- Most busy doctor
- Hospital load
- Emergency trends

## PySpark ML Risk Prediction

Large datasets can be trained through Spark ML after they are exported to HDFS.

Upload the sample patient risk training data:

```bash
docker cp analytics_samples/patients.csv healthcare_hdfs_namenode:/tmp/patients.csv
docker compose --env-file .env.production -f docker-compose.prod.yml --profile analytics exec -T namenode hdfs dfs -mkdir -p /healthcare
docker compose --env-file .env.production -f docker-compose.prod.yml --profile analytics exec -T namenode hdfs dfs -put -f /tmp/patients.csv /healthcare/patients.csv
```

Train a Spark Random Forest model through the API:

```bash
curl -X POST https://your-domain.example.com/ml/risk/train
```

Predict patient risk through the API:

```bash
curl -X POST https://your-domain.example.com/ml/risk/predict \
  -H "Content-Type: application/json" \
  -d '{"age":64,"chest_pain":true,"diabetes":true,"hypertension":true,"shortness_of_breath":true,"blood_pressure":158}'
```

You can also run the standalone Spark scripts inside the Spark master container:

```bash
docker cp analytics_samples/spark_train_risk_model.py healthcare_spark_master:/tmp/spark_train_risk_model.py
docker compose --env-file .env.production -f docker-compose.prod.yml --profile analytics exec -T spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /tmp/spark_train_risk_model.py

docker cp analytics_samples/spark_predict_risk.py healthcare_spark_master:/tmp/spark_predict_risk.py
docker compose --env-file .env.production -f docker-compose.prod.yml --profile analytics exec -T spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /tmp/spark_predict_risk.py
```

## Service Map

Only Nginx is public:

| Service | Network | Purpose |
|---|---|---|
| `nginx` | Public `80/443` | HTTPS termination and reverse proxy |
| `main_app` | Internal `8000` | FastAPI browser app and API |
| `mcp_server` | Internal `8001` | MCP tools and healthcare rules |
| `postgres` | Internal `5432` | Persistent application database |
| `namenode` | Optional analytics profile | HDFS metadata service |
| `datanode` | Optional analytics profile | HDFS data storage |
| `spark-master` | Optional analytics profile | Spark cluster coordinator |
| `spark-worker` | Optional analytics profile | Spark job execution |
