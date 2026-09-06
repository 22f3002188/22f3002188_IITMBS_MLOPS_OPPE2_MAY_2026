
# IITM BS MLOps OPPE-2 — Heart Disease Prediction API

## Project Overview

This project implements an end-to-end MLOps pipeline for a heart disease classification model.

The project covers:

1. Private GitHub repository
2. Model explainability using SHAP
3. Fairness analysis using Fairlearn
4. Dockerized FastAPI application deployed on Google Kubernetes Engine (GKE)
5. Horizontal Pod Autoscaling (HPA)
6. Automated CI/CD using GitHub Actions and Google Cloud Workload Identity Federation
7. API prediction logging using Google Cloud Logging
8. 100 prediction requests through the deployed API
9. Load testing using `wrk` with more than 2,000 concurrent connections
10. Input data drift detection using KS statistical testing and Evidently

---

## 1. Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── app/
│   ├── main.py
│   ├── train_model.py
│   └── model/
│       └── model.joblib
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
│
├── data.csv
├── requirements.txt
├── Dockerfile
│
├── explainability.py
├── shap_feature_importance.csv
├── shap_summary.png
│
├── fairness.py
├── fairlearn_age_group_metrics.csv
├── fairness_summary.csv
│
├── prediction_data_100.csv
├── task5_api_predictions.csv
│
├── load_test.lua
│
├── drift_detection.py
├── drift_summary.csv
└── drift_report.html

2. Dataset

The project uses data.csv for heart disease classification.

The dataset contains:

303 observations
15 columns
Columns
sno
age
gender
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
target

The sno column is an observation identifier and is excluded from model training.

The target column is the prediction target.

Target distribution:

yes = 165
no  = 138
3. Model Training

The model is implemented using a preprocessing pipeline and Random Forest classifier.

Preprocessing
Missing numerical values are imputed using the median.
Missing categorical values are imputed using the most frequent value.
gender is one-hot encoded.
sno is removed.
target is separated as the prediction variable.
Train/Test Split
test_size = 0.20
random_state = 42
stratify = target
Model
RandomForestClassifier

n_estimators = 300
random_state = 42
class_weight = balanced
Model Result

Test accuracy:

0.8033

The trained preprocessing pipeline and model are stored in:

app/model/model.joblib
4. Task 2 — Model Explainability using SHAP

SHAP (SHapley Additive exPlanations) is used to understand the contribution of features to model predictions.

Implementation:

explainability.py

Generated outputs:

shap_feature_importance.csv
shap_summary.png
Least Impactful Features

Based on mean absolute SHAP values:

Feature	Mean Absolute SHAP
fbs	0.001902
trestbps	0.013237
restecg	0.015150
Plain-English Interpretation

Among the analyzed features, fbs had the smallest average contribution to the model predictions.

trestbps and restecg also had relatively low average SHAP impact compared with the more influential features.

5. Task 3 — Fairness Analysis using Fairlearn

Fairness analysis is implemented in:

fairness.py

The sensitive attribute used for the fairness analysis is:

age

Age was divided into three groups:

<50
50-59
60+
Group Sizes
Age Group	Samples
<50	88
50-59	125
60+	90
Overall Accuracy
0.8033
Group Metrics
Age Group	Accuracy	Selection Rate
<50	0.937500	0.812500
50-59	0.740741	0.629630
60+	0.777778	0.500000
Fairness Metrics
Demographic Parity Difference = 0.3125
Equalized Odds Difference      = 0.1429

These results indicate measurable differences in model behavior across the defined age groups.

Generated files:

fairlearn_age_group_metrics.csv
fairness_summary.csv
6. Task 4 — FastAPI Application

The prediction API is implemented using FastAPI.

Main application:

app/main.py
Health Endpoint
GET /health

Example response:

{
  "status": "healthy",
  "model_loaded": true
}
Prediction Endpoint
POST /predict

The endpoint accepts:

age
gender
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
Example Request
{
  "age": 63,
  "gender": "male",
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
Example Response
{
  "prediction": "yes",
  "heart_disease_probability": 0.586667,
  "timestamp": "2026-09-06T09:15:36.152974+00:00"
}

The application logs prediction requests containing:

timestamp
input
prediction
probability
7. Dockerization

The FastAPI application is containerized using Docker.

Docker configuration:

Dockerfile

The container:

Uses Python 3.12
Installs dependencies from requirements.txt
Copies the application
Exposes port 8000
Runs FastAPI using Uvicorn
Build Locally
docker build -t heart-disease-api:1.0 .
Run Locally
docker run -p 8000:8000 heart-disease-api:1.0

Health check:

curl http://localhost:8000/health
8. Google Artifact Registry

The Docker image is stored in Google Artifact Registry.

Configuration:

Project:
project-56d2d662-7634-434f-a2a

Region:
us-central1

Repository:
iris-repo

Image:
heart-disease-api

CI/CD images are tagged using the GitHub commit SHA.

9. Google Kubernetes Engine

The application is deployed to Google Kubernetes Engine.

Cluster:

heart-disease-cluster

Region:

us-central1

Kubernetes configuration:

k8s/
Deployment
k8s/deployment.yaml

The deployment contains:

FastAPI container
CPU and memory requests/limits
Startup probe
Readiness probe
Liveness probe

The API container listens on:

8000

All health probes use:

/health
10. Kubernetes Service

The API is exposed using a Kubernetes LoadBalancer.

Configuration:

k8s/service.yaml

The external endpoint used during testing was:

http://34.172.95.252

Health check:

curl http://34.172.95.252/health

Result:

{
  "status": "healthy",
  "model_loaded": true
}
11. Horizontal Pod Autoscaling

HPA configuration:

k8s/hpa.yaml

Configuration:

Minimum replicas: 1
Maximum replicas: 3
CPU target: 60%
Observed Scaling

During the preliminary load test:

CPU: 125% / 60%
Replicas: 2

During the main high-concurrency test:

CPU: 188% / 60%
Replicas: 2

This demonstrates that the Kubernetes HPA reacted to increased CPU utilization and scaled the application.

12. GitHub Actions CI/CD

CI/CD workflow:

.github/workflows/deploy.yml

The workflow runs whenever changes are pushed to:

main
CI/CD Pipeline
GitHub Repository
       |
       v
GitHub Actions
       |
       v
Workload Identity Federation
       |
       v
Google Cloud Authentication
       |
       v
Docker Build
       |
       v
Artifact Registry
       |
       v
GKE Credentials
       |
       v
Kubernetes Deployment Update
       |
       v
Rollout Verification

Workload Identity Federation is used instead of storing a long-lived Google Cloud service-account key in GitHub.

Successful Workflow

Workflow:

Build and Deploy Heart Disease API

Successful commit:

4945b77

Result:

Success

Duration:

5 minutes 6 seconds

The workflow successfully:

Authenticated to Google Cloud.
Built the Docker image.
Pushed the image to Artifact Registry.
Obtained GKE credentials.
Updated the Kubernetes deployment.
Waited for rollout completion.
Verified the deployment, pods and HPA.
13. Task 5 — 100 API Predictions and Cloud Logging

A 100-row prediction dataset was generated:

prediction_data_100.csv

The dataset contains exactly:

100 prediction rows

All 100 rows were sent to the deployed GKE API.

API Results
Total requests: 100
Successful: 100
Failed: 0

The prediction results were saved to:

task5_api_predictions.csv

The output contains:

row_id
request_timestamp
input
prediction
heart_disease_probability
api_timestamp
status

The file contains:

101 lines

which means:

1 header + 100 prediction records
Cloud Logging

The FastAPI application prediction logs were verified through Google Cloud Logging.

The logs contain:

prediction_request
timestamp
input
prediction
probability

The Cloud Logging query returned:

100

prediction log entries corresponding to the 100 API requests.

End-to-End Verification
100 input rows
       |
       v
100 API requests
       |
       v
100 successful predictions
       |
       v
100 prediction records
       |
       v
100 Cloud Logging entries
14. Task 6 — Load Testing with wrk

The deployed API was load tested using:

wrk 4.1.0

A Lua script was used to generate POST requests:

load_test.lua
High-Concurrency Test

The required test used:

Threads: 4
Connections: 2001
Duration: 30 seconds
Timeout: 10 seconds

Command:

wrk -t4 -c2001 -d30s --latency --timeout 10s -s load_test.lua http://34.172.95.252/predict
Results

Total completed requests:

45 requests in 30.05s

Throughput:

1.50 requests/sec

Average latency:

9.29 seconds

Maximum latency:

9.98 seconds

Latency distribution:

50% = 9.98s
75% = 9.98s
90% = 9.98s
99% = 9.98s

Socket errors:

connect = 0
read    = 225
write   = 13631
timeout = 41
Load-Test Analysis

The API remained reachable under 2,001 concurrent connections because there were zero connection errors.

However, the service became heavily saturated.

Average latency increased to 9.29 seconds and most completed requests approached the 10-second timeout.

The low throughput of 1.50 requests/sec together with read, write and timeout errors demonstrates that the API could not efficiently handle this extreme concurrency level.

The Kubernetes HPA reacted to increased CPU utilization and scaled the deployment from one replica to two replicas.

Therefore, the test demonstrates both:

performance degradation under extreme concurrency
Kubernetes horizontal autoscaling behavior
15. Task 7 — Input Drift Detection

Input drift detection is implemented in:

drift_detection.py

The script compares the training/reference distribution against the generated 100-row prediction dataset.

Reference data:

data.csv

Current/prediction data:

prediction_data_100.csv

Exactly 100 prediction rows are generated.

Statistical Test

The Kolmogorov-Smirnov two-sample test is applied to:

age
trestbps
chol
thalach
oldpeak

Drift is considered detected when:

p-value < 0.05
Results
Feature	KS Statistic	p-value	Drift Detected
age	0.197294	0.004758	Yes
trestbps	0.352609	<0.000001	Yes
chol	0.537152	<0.000001	Yes
thalach	0.288725	0.000005	Yes
oldpeak	0.505281	<0.000001	Yes

The largest observed distribution differences were for:

chol
oldpeak
Evidently

The project also generates an Evidently DataDriftPreset report.

Generated files:

prediction_data_100.csv
drift_summary.csv
drift_report.html

The HTML report provides an additional visual/data-drift analysis of the reference and current datasets.

16. Requirements

Python dependencies are listed in:

requirements.txt

Major packages include:

numpy
pandas
scikit-learn
matplotlib
seaborn
mlflow
shap
lime
evidently
fastapi
uvicorn
joblib
fairlearn
17. Running the Application Locally

Activate the virtual environment:

source venv/bin/activate

Train/save the model:

python app/train_model.py

Start the API:

uvicorn app.main:app --host 0.0.0.0 --port 8000

Health check:

curl http://localhost:8000/health

Prediction endpoint:

POST http://localhost:8000/predict
18. Running Explainability
python explainability.py

Outputs:

shap_feature_importance.csv
shap_summary.png
19. Running Fairness Analysis
python fairness.py

Outputs:

fairlearn_age_group_metrics.csv
fairness_summary.csv
20. Running Drift Detection
python drift_detection.py

Outputs:

prediction_data_100.csv
drift_summary.csv
drift_report.html
21. Kubernetes Commands

Apply the deployment:

kubectl apply -f k8s/deployment.yaml

Apply the service:

kubectl apply -f k8s/service.yaml

Apply the HPA:

kubectl apply -f k8s/hpa.yaml

Check deployment:

kubectl get deployment heart-disease-api

Check pods:

kubectl get pods

Check service:

kubectl get service heart-disease-api

Check HPA:

kubectl get hpa heart-disease-api

Check resource usage:

kubectl top pods
22. GitHub Actions

The CI/CD workflow is automatically triggered by a push to the main branch.

The workflow file is:

.github/workflows/deploy.yml

The workflow uses:

GitHub OIDC
Workload Identity Federation
Google Cloud
Artifact Registry
Google Kubernetes Engine
kubectl

No long-lived Google Cloud service-account key is stored in the GitHub repository.

23. Project Completion Summary
Task	Status
Private GitHub repository	Complete
SHAP explainability	Complete
Fairlearn fairness analysis	Complete
Dockerized FastAPI	Complete
Artifact Registry	Complete
GKE deployment	Complete
HPA with maximum 3 pods	Complete
GitHub Actions CI/CD	Complete
100 API predictions	Complete
Input/output/timestamp logging	Complete
Cloud Logging verification	Complete
2,001 concurrent connection test	Complete
Load-test analysis	Complete
Input drift detection	Complete
KS statistical drift analysis	Complete
Evidently drift report	Complete
24. Final Architecture
                         GitHub Repository
                                |
                                v
                         GitHub Actions
                                |
                                v
                    Workload Identity Federation
                                |
                                v
                         Google Cloud
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
          Artifact Registry                  GKE
                 |                             |
           Docker Image              Kubernetes Deployment
                                               |
                                    +----------+----------+
                                    |                     |
                                    v                     v
                                  Pod 1              Pod 2 / Pod 3
                                    |                     |
                                    +----------+----------+
                                               |
                                               v
                                          LoadBalancer
                                               |
                                               v
                                        FastAPI API
                                               |
                              +----------------+----------------+
                              |                                 |
                              v                                 v
                         Prediction                         Cloud Logging
                              |
                              v
                       Prediction Results

Training Data
     |
     +--> Model Training
     |
     +--> SHAP Explainability
     |
     +--> Fairness Analysis
     |
     +--> Drift Reference Distribution
                       |
                       v
              100-Row Prediction Data
                       |
                       v
                Drift Detection
                 /            \
                v              v
         KS Statistical     Evidently
             Test             Report
25. Final Conclusion

This project demonstrates an end-to-end MLOps workflow for a heart disease classification model.

The implementation covers:

model training
preprocessing
SHAP explainability
fairness evaluation with Fairlearn
Docker containerization
Artifact Registry
Google Kubernetes Engine deployment
Kubernetes Service
Horizontal Pod Autoscaling
GitHub Actions CI/CD
Workload Identity Federation
API prediction logging
Google Cloud Logging
100 deployed API predictions
high-concurrency load testing
performance analysis
input drift detection
Evidently monitoring

The deployed API was successfully accessed through GKE.

All 100 prediction requests completed successfully and were verified in Cloud Logging.

The HPA successfully responded to increased CPU utilization during load testing.

The 2,001-concurrent-connection benchmark demonstrated significant latency and error behavior under extreme load, providing evidence for performance analysis.

Input drift was evaluated using both the Kolmogorov-Smirnov statistical test and an Evidently data-drift report.

Overall, the project demonstrates the complete lifecycle from model development to deployment, monitoring, testing, fairness, explainability, CI/CD and drift detection.
EOF