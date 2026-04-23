import requests

FHIR_BASE = "https://r4.smarthealthit.org"


def _get_json(url, params=None):
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_patients(limit=5):
    data = _get_json(f"{FHIR_BASE}/Patient", params={"_count": limit})
    return data.get("entry", [])


def get_patient(patient_id):
    return _get_json(f"{FHIR_BASE}/Patient/{patient_id}")


def get_observations(patient_id):
    data = _get_json(f"{FHIR_BASE}/Observation", params={"patient": patient_id, "_count": 200})
    return data.get("entry", [])