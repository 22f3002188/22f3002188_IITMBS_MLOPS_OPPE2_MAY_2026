# MLOps OPPE-2 — Heart Disease Classification

## 1. Project Overview

This project implements an end-to-end MLOps workflow for a **Heart Disease Classification** machine learning application.

The project covers the complete lifecycle of a machine learning application:

**Data → Preprocessing → Model Training → Explainability → Fairness → API → Docker → Artifact Registry → GKE → HPA → CI/CD → Logging → Load Testing → Drift Detection**

The machine learning model predicts whether a patient is likely to have heart disease based on clinical input features.

The project was developed and deployed using **Google Cloud Platform (GCP)**, with the application containerized using Docker and deployed to **Google Kubernetes Engine (GKE)**.

---

# 2. Problem Statement

The objective of this project is to develop a machine learning classification system that predicts whether a patient has heart disease.

This is a **supervised binary classification problem**.

The dataset contains patient-related features such as:

- Age
- Gender
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- Oldpeak
- Slope
- Number of major vessels
- Thalassemia-related information

The target variable indicates whether heart disease is present or not.

The goal was not only to train a machine learning model, but also to build a complete and deployable MLOps solution with:

- Explainability
- Fairness analysis
- REST API
- Containerization
- Cloud deployment
- Kubernetes orchestration
- Autoscaling
- CI/CD
- Cloud logging
- Load testing
- Input drift detection

---

# 3. Dataset

The project uses the `data.csv` dataset.

Dataset shape:

```text
303 rows × 15 columns

Columns:
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

The target distribution is:
yes    165
no     138

The sno column is an identifier and was excluded from model training.

The target column was converted into a binary representation:

yes → 1
no  → 0

4. Project Structure
22f3002188_IITMBS_MLOPS_OPPE2_MAY_2026/
│
├── app/
│   ├── model/
│   │   └── model.joblib
│   │
│   ├── main.py
│   └── train_model.py
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── data.csv
├── explainability.py
├── fairness.py
├── drift_detection.py
├── task5_predictions.py
├── task5_api_predictions.csv
├── prediction_data_100.csv
├── load_test.lua
├── shap_feature_importance.csv
├── shap_summary.png
├── drift_summary.csv
├── drift_report.html
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md

5. Technology Stack

The project uses the following technologies:
| Component            | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python 3.12                  |
| ML Model             | Random Forest Classifier     |
| Data Processing      | Pandas, Scikit-learn         |
| Explainability       | SHAP                         |
| Fairness             | Fairlearn                    |
| API                  | FastAPI                      |
| API Server           | Uvicorn                      |
| Containerization     | Docker                       |
| Container Registry   | Google Artifact Registry     |
| Cloud Platform       | Google Cloud Platform        |
| Kubernetes           | Google Kubernetes Engine     |
| Autoscaling          | Kubernetes HPA               |
| CI/CD                | GitHub Actions               |
| Authentication       | Workload Identity Federation |
| Logging              | Google Cloud Logging         |
| Load Testing         | wrk                          |
| Drift Detection      | SciPy + Evidently            |
| Version Control      | Git + GitHub                 |

6. Machine Learning Approach
6.1 Data Preprocessing

The preprocessing pipeline performs the following operations:

Remove the sno identifier column.
Separate the target variable from input features.
Handle missing numerical values using median imputation.
Handle missing categorical values using most-frequent imputation.
Apply one-hot encoding to categorical variables.
Perform a stratified train-test split.
Train the classification model.

The train-test split uses:

Test size = 20%
Random state = 42
Stratification = Enabled

This ensures reproducibility and maintains the target class distribution between training and testing datasets.

7. Machine Learning Model

A Random Forest Classifier was used.

Configuration:

n_estimators = 300
random_state = 42
class_weight = balanced

The model achieved:

Test Accuracy = 0.8033

Therefore, the model achieved approximately:

80.33% test accuracy

The trained model and preprocessing pipeline were saved into:

app/model/model.joblib
8. Task 2 — Explainability Using SHAP

Model explainability was implemented using SHAP (SHapley Additive exPlanations).

The purpose of SHAP is to understand how individual features contribute to the model's predictions.

The script:

explainability.py

performs:

Data preprocessing
Model training
SHAP TreeExplainer analysis
Feature importance calculation
SHAP summary plot generation

Output files:

shap_feature_importance.csv
shap_summary.png

The least-impact features based on mean absolute SHAP values were:

Feature	Mean Absolute SHAP
fbs	0.001902
trestbps	0.013237
restecg	0.015150

Among these analysed features, fbs had the lowest mean absolute SHAP impact.

9. Task 3 — Fairness Analysis

Fairness analysis was performed using Fairlearn.

The sensitive attribute selected for the analysis was:

age

The age values were grouped into:

<50
50–59
60+

Group sizes:

Age Group	Number of Records
<50	88
50–59	125
60+	90

Overall model accuracy:

0.8033

Group metrics:

Age Group	Accuracy	Selection Rate
50–59	0.740741	0.629630
60+	0.777778	0.500000
<50	0.937500	0.812500

Fairness metrics:

Demographic Parity Difference = 0.3125

Equalized Odds Difference ≈ 0.1429

These values indicate measurable differences between the age groups.

The fairness analysis demonstrates that model performance and prediction selection rates can differ across sensitive demographic groups.

10. FastAPI Application

The trained model was converted into a REST API using FastAPI.

The API code is located at:

app/main.py

The API loads:

app/model/model.joblib

The API provides two main endpoints.

10.1 Health Endpoint

Endpoint:

GET /health

Example:

curl http://<EXTERNAL_IP>/health

Expected response:

{
  "status": "healthy",
  "model_loaded": true
}

This endpoint confirms that:

The API is running.
The trained model has been successfully loaded.
10.2 Prediction Endpoint

Endpoint:

POST /predict

The endpoint accepts patient information including:

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

Example request:

curl -X POST http://<EXTERNAL_IP>/predict \
-H "Content-Type: application/json" \
-d '{"age":54.0,"gender":"female","cp":2,"trestbps":124.0,"chol":199.0,"fbs":0,"restecg":0,"thalach":167.0,"exang":0,"oldpeak":2.4,"slope":1,"ca":0,"thal":3}'

Example response:

{
  "prediction": "yes",
  "heart_disease_probability": 0.586667,
  "timestamp": "2026-09-06T09:15:36.152974+00:00"
}

The API returns:

Predicted class
Heart disease probability
UTC timestamp
11. API Logging

The FastAPI application uses INFO-level application logging.

Prediction requests are logged with information including:

Input data
Prediction
Probability
Timestamp

Example log information:

prediction_request
input
prediction
probability
timestamp

These logs are collected in Google Cloud Logging after deployment to GKE.

12. Docker Containerization

The application was containerized using Docker.

The Dockerfile uses:

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

Docker image:

heart-disease-api:1.0

The image was tested locally before cloud deployment.

Local health test:

curl http://localhost:8000/health

The container successfully returned:

{
  "status": "healthy",
  "model_loaded": true
}
13. Google Artifact Registry

The Docker image was pushed to Google Artifact Registry.

Configuration:

Project:
project-56d2d662-7634-434f-a2a

Region:
us-central1

Repository:
iris-repo

Image:
heart-disease-api

Artifact Registry image:

us-central1-docker.pkg.dev/project-56d2d662-7634-434f-a2a/iris-repo/heart-disease-api

The image was successfully built and pushed to Artifact Registry.

14. Google Kubernetes Engine

The application was deployed to Google Kubernetes Engine.

Cluster:

heart-disease-cluster

Region:

us-central1

Cluster type:

GKE Autopilot

The Kubernetes configuration contains:

k8s/deployment.yaml
k8s/service.yaml
k8s/hpa.yaml
15. Kubernetes Deployment

The application is deployed using a Kubernetes Deployment.

Deployment name:

heart-disease-api

The container exposes port:

8000

The deployment includes:

CPU and memory requests
CPU and memory limits
Startup probe
Readiness probe
Liveness probe

The health endpoint is used for Kubernetes health checks:

/health

This allows Kubernetes to determine whether the application is ready and healthy.

16. Kubernetes LoadBalancer Service

A Kubernetes LoadBalancer Service was configured.

Service:

heart-disease-api

The service exposes:

Port 80

and forwards traffic to:

Container Port 8000

The service received an external IP:

34.172.95.252

The deployed API was successfully accessed through this external IP.

Example:

curl http://34.172.95.252/health

Response:

{
  "status": "healthy",
  "model_loaded": true
}
17. Horizontal Pod Autoscaler

A Kubernetes Horizontal Pod Autoscaler was configured.

Configuration:

Minimum replicas = 1
Maximum replicas = 3
CPU target = 60%

The HPA monitors CPU utilization.

Configuration:

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: heart-disease-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: heart-disease-api
  minReplicas: 1
  maxReplicas: 3

During load testing, CPU utilization increased and the HPA scaled the application from:

1 replica → 2 replicas

This demonstrated Kubernetes autoscaling behaviour.

18. GitHub Actions CI/CD

CI/CD was implemented using GitHub Actions.

Workflow file:

.github/workflows/deploy.yml

The workflow is triggered by:

push to main branch

The workflow performs the following operations:

Checkout source code.
Authenticate to Google Cloud.
Use Workload Identity Federation.
Configure Docker authentication.
Build the Docker image.
Push the Docker image to Artifact Registry.
Obtain GKE credentials.
Update the Kubernetes Deployment.
Wait for rollout completion.
Verify deployment, pods and HPA.

The GitHub Actions deployment workflow completed successfully.

19. Workload Identity Federation

GitHub Actions authentication was configured using Workload Identity Federation.

The configuration includes:

Workload Identity Pool:
github-actions-pool

Provider:

github-actions

GitHub issuer:

https://token.actions.githubusercontent.com/

Repository restriction:

22f3002188/22f3002188_IITMBS_MLOPS_OPPE2_MAY_2026

The repository restriction prevents unrelated GitHub repositories from using the configured identity provider.

This avoids storing a long-lived Google Cloud service account key inside GitHub Actions.

20. Task 5 — 100 API Predictions

A separate script was created:

task5_predictions.py

The script sends 100 records from:

prediction_data_100.csv

to:

/predict

The output is saved as:

task5_api_predictions.csv

Results:

Total requests: 100
Successful: 100
Failed: 0

Therefore, all 100 API prediction requests were successfully processed.

The output CSV contains:

row_id
request_timestamp
input
prediction
heart_disease_probability
api_timestamp
status
21. Google Cloud Logging

The prediction requests were verified using Google Cloud Logging.

A Kubernetes container log query was used to search for prediction request entries.

The logs contained information such as:

prediction_request
input
prediction
probability
timestamp

The final verification returned:

100 prediction_request log entries

This demonstrates application observability in the deployed GCP environment.

22. Task 6 — Load Testing Using wrk

Load testing was performed using:

wrk

The request configuration was stored in:

load_test.lua

The test used:

Threads = 4
Connections = 2001
Duration = 30 seconds
Timeout = 10 seconds

Command:

wrk -t4 -c2001 -d30s --latency --timeout 10s \
-s load_test.lua \
http://34.172.95.252/predict

Observed results:

Duration ≈ 30.05 seconds
Requests completed = 45
Throughput ≈ 1.50 requests/sec
Average latency ≈ 9.29 seconds
Maximum latency ≈ 9.98 seconds

Socket errors included:

Connect errors = 0
Read errors = 225
Write errors = 13631
Timeout errors = 41

The test intentionally generated a very high level of concurrent traffic.

The important observation was that the service remained reachable, but the application became heavily saturated.

23. HPA Response During Load Testing

Before the main load test, the application normally ran with:

1 replica

During the high-concurrency load test:

CPU utilization increased significantly

The HPA reacted by scaling:

1 replica → 2 replicas

This demonstrated that the Kubernetes Horizontal Pod Autoscaler was responding to increased CPU utilization.

After the workload reduced, the deployment was able to settle back toward its normal replica count.

24. Task 7 — Input Drift Detection

Input drift detection was implemented using:

drift_detection.py

The script compares the training data distribution with the generated prediction input data.

The numerical features analysed were:

age
trestbps
chol
thalach
oldpeak

The statistical test used was:

Kolmogorov-Smirnov two-sample test

The significance threshold was:

p < 0.05

A feature was considered to show statistically significant drift when:

p-value < 0.05
25. Drift Detection Results

The observed results were:

Feature	KS Statistic	p-value	Drift
age	0.197294	0.004758	True
trestbps	0.352609	0.0	True
chol	0.537152	0.0	True
thalach	0.288725	0.000005	True
oldpeak	0.505281	0.0	True

All five analysed numerical features showed statistically significant drift according to the selected threshold.

The generated data was sampled using a different distribution-generation method, so the observed drift is expected from the way the test input data was constructed.

26. Evidently Drift Report

In addition to the statistical test, Evidently was used to generate a visual drift report.

Output:

drift_report.html

A summary CSV was also generated:

drift_summary.csv

The Evidently report provides an additional way to inspect differences between reference training data and current prediction input data.

27. Important Output Files

The major output files generated by the project are:

File	Purpose
shap_feature_importance.csv	SHAP feature importance
shap_summary.png	SHAP summary visualization
model.joblib	Trained deployable model
prediction_data_100.csv	100 API input records
task5_api_predictions.csv	100 API prediction results
drift_summary.csv	Statistical drift results
drift_report.html	Evidently drift report
28. Errors Encountered and Troubleshooting

Several issues were encountered during implementation.

28.1 Cloud Logging Permission Error

Initially, querying Google Cloud Logging resulted in a permissions issue.

The Compute Engine service account did not have the required Logs Viewer permission.

The issue was identified as an IAM permission problem.

After the appropriate Logs Viewer permission was added, Cloud Logging queries worked successfully.

28.2 IAM Permission Error

While attempting to grant read-only access to the IITM evaluation account, the following type of error was encountered:

does not have permission to access projects instance

The reason was that the command was executed while authenticated as:

601911737889-compute@developer.gserviceaccount.com

The Compute Engine service account did not have permission to modify the project's IAM policy.

Therefore, IAM modifications must be performed using an account with appropriate project-level IAM administration permissions.

This was identified as an IAM authorization issue rather than an application or deployment problem.

28.3 High-Concurrency Load Test Saturation

During the 2,001-connection load test, the service became heavily saturated.

This caused:

High latency
Read errors
Write errors
Request timeouts
Low throughput

However, there were:

0 connection errors

and the HPA reacted by increasing the number of replicas.

This demonstrated the behaviour of the application under stress and the response of Kubernetes autoscaling.

29. End-to-End Architecture

The complete architecture can be represented as:

                    ┌─────────────────┐
                    │    data.csv     │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Data Preprocessing  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Random Forest Model │
                  └──────────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
          ┌──────────────┐        ┌──────────────┐
          │     SHAP     │        │   Fairlearn  │
          │ Explainability│       │   Fairness   │
          └──────────────┘        └──────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │     FastAPI         │
                  │   /health /predict  │
                  └──────────┬──────────┘
                             │
                             ▼
                       ┌───────────┐
                       │  Docker   │
                       └─────┬─────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Artifact Registry   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │        GKE          │
                  │  heart-disease-api  │
                  └──────────┬──────────┘
                             │
                     ┌───────┴────────┐
                     │                │
                     ▼                ▼
              LoadBalancer           HPA
                     │                │
                     ▼                ▼
              Public API        1 → 2 replicas
                     │
          ┌──────────┼───────────┐
          │          │           │
          ▼          ▼           ▼
      100 API     Cloud       Load Test
     Predictions  Logging       wrk
          │
          ▼
     Drift Detection
          │
          ▼
   Evidently Report

30. End-to-End Workflow

The complete sequence performed in the project was:

1. Create/access GCP Compute Engine VM
        ↓
2. SSH into VM
        ↓
3. Configure Python environment
        ↓
4. Install required dependencies
        ↓
5. Configure Git and GitHub repository
        ↓
6. Load and analyse dataset
        ↓
7. Preprocess data
        ↓
8. Train Random Forest model
        ↓
9. Evaluate model
        ↓
10. Perform SHAP explainability
        ↓
11. Perform Fairlearn fairness analysis
        ↓
12. Create FastAPI application
        ↓
13. Save trained model as model.joblib
        ↓
14. Build Docker image
        ↓
15. Test Docker container locally
        ↓
16. Push image to Artifact Registry
        ↓
17. Create GKE Autopilot cluster
        ↓
18. Deploy application to GKE
        ↓
19. Configure LoadBalancer Service
        ↓
20. Configure HPA
        ↓
21. Configure GitHub Actions CI/CD
        ↓
22. Configure Workload Identity Federation
        ↓
23. Deploy through GitHub Actions
        ↓
24. Test /health endpoint
        ↓
25. Test /predict endpoint
        ↓
26. Send 100 prediction requests
        ↓
27. Verify Cloud Logging
        ↓
28. Perform wrk load testing
        ↓
29. Observe HPA scaling
        ↓
30. Perform input drift detection
        ↓
31. Generate drift_summary.csv
        ↓
32. Generate Evidently drift_report.html

31. Demonstration Commands

Useful commands for demonstrating the deployed system:

Check GCP authentication
gcloud auth list
Check GKE cluster
gcloud container clusters list
Check Kubernetes pods
kubectl get pods
Check deployment
kubectl get deployment
Check service
kubectl get service
Check HPA
kubectl get hpa
Health check
curl http://34.172.95.252/health
Prediction
curl -X POST http://34.172.95.252/predict \
-H "Content-Type: application/json" \
-d '{"age":54.0,"gender":"female","cp":2,"trestbps":124.0,"chol":199.0,"fbs":0,"restecg":0,"thalach":167.0,"exang":0,"oldpeak":2.4,"slope":1,"ca":0,"thal":3}'
Run 100 API predictions
python task5_predictions.py
Run drift detection
python drift_detection.py
32. Results Summary
Component	Result
Dataset	303 × 15
Model	Random Forest
Test Accuracy	80.33%
SHAP	Successfully implemented
Fairness	Successfully implemented
Demographic Parity Difference	0.3125
Equalized Odds Difference	≈ 0.1429
FastAPI	Successfully deployed
Docker	Successfully built and tested
Artifact Registry	Successfully pushed
GKE	Successfully deployed
LoadBalancer	Successfully exposed
HPA	Successfully configured
CI/CD	Successfully executed
API Predictions	100/100 successful
Cloud Logging	100 prediction entries verified
Load Test	2,001 concurrent connections
Load Test Throughput	≈ 1.50 req/sec
Load Test Average Latency	≈ 9.29 sec
HPA Load Response	1 → 2 replicas
Drift Detection	Successfully implemented
Evidently Report	Successfully generated
33. MLOps Practices Demonstrated

This project demonstrates the following MLOps practices:

Reproducibility

Fixed random states were used during data splitting and model training.

Explainability

SHAP was used to understand feature contributions.

Fairness

Fairlearn was used to evaluate differences across age groups.

Packaging

The application and model were packaged into a Docker container.

Registry

The container image was stored in Google Artifact Registry.

Deployment

The application was deployed to GKE.

Scalability

Horizontal Pod Autoscaling was configured.

CI/CD

GitHub Actions automated the build and deployment process.

Authentication

Workload Identity Federation was used instead of storing long-lived service account keys in GitHub.

Monitoring

Application prediction logs were verified through Cloud Logging.

Performance Testing

The API was tested using wrk with 2,001 concurrent connections.

Data Monitoring

Input drift was evaluated using statistical testing and Evidently.

34. Conclusion

This project demonstrates a complete end-to-end MLOps lifecycle for a heart disease classification application.

The workflow started with the dataset and machine learning model and progressed through:

Data
→ Model
→ Explainability
→ Fairness
→ FastAPI
→ Docker
→ Artifact Registry
→ GKE
→ HPA
→ CI/CD
→ Cloud Logging
→ Load Testing
→ Drift Detection

The final system successfully provides a cloud-hosted machine learning prediction API.

The project also demonstrates important production-oriented MLOps concepts including automated deployment, containerization, Kubernetes orchestration, autoscaling, logging, explainability, fairness evaluation, performance testing and input drift detection.

35. Project Status
Task 1 — Private GitHub Repository              ✅ Complete
Task 2 — SHAP Explainability                   ✅ Complete
Task 3 — Fairness Analysis                     ✅ Complete
Task 4 — FastAPI + Docker + GKE + CI/CD        ✅ Complete
Task 5 — 100 API Predictions + Logging         ✅ Complete
Task 6 — wrk Load Testing + HPA                ✅ Complete
Task 7 — Input Drift Detection                ✅ Complete
Documentation                                  ✅ Complete
36. Final Statement

The project successfully demonstrates an end-to-end MLOps implementation for a machine learning classification application using Google Cloud Platform.

The system integrates machine learning, explainability, fairness, API serving, containerization, Kubernetes deployment, autoscaling, CI/CD, logging, performance testing and data drift monitoring into a single workflow.
