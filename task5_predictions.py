import csv
import json
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

API_URL = "http://34.172.95.252/predict"
INPUT_FILE = "prediction_data_100.csv"
OUTPUT_FILE = "task5_api_predictions.csv"

with open(INPUT_FILE, newline="") as f:
    rows = list(csv.DictReader(f))

results = []

for i, row in enumerate(rows, start=1):
    payload = {}

    for key, value in row.items():
        if key == "gender":
            payload[key] = value
        elif key in {"cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}:
            payload[key] = int(float(value))
        else:
            payload[key] = float(value)

    request_timestamp = datetime.now(timezone.utc).isoformat()

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        results.append({
            "row_id": i,
            "request_timestamp": request_timestamp,
            "input": json.dumps(payload, separators=(",", ":")),
            "prediction": response_data["prediction"],
            "heart_disease_probability": response_data["heart_disease_probability"],
            "api_timestamp": response_data["timestamp"],
            "status": "success",
        })

        print(
            f"{i:03d}/100 | "
            f"prediction={response_data['prediction']} | "
            f"probability={response_data['heart_disease_probability']}"
        )

    except Exception as e:
        results.append({
            "row_id": i,
            "request_timestamp": request_timestamp,
            "input": json.dumps(payload, separators=(",", ":")),
            "prediction": "",
            "heart_disease_probability": "",
            "api_timestamp": "",
            "status": f"error: {e}",
        })

        print(f"{i:03d}/100 | ERROR | {e}")

    time.sleep(0.05)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "row_id",
            "request_timestamp",
            "input",
            "prediction",
            "heart_disease_probability",
            "api_timestamp",
            "status",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

successful = sum(r["status"] == "success" for r in results)
failed = len(results) - successful

print()
print(f"Total requests: {len(results)}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Output: {OUTPUT_FILE}")
