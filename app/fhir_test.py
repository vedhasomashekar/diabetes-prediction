from pprint import pprint

from fhir_client import get_observations, get_patient, get_patients
from fhir_mapper import map_fhir_to_features
from model_adapter import prepare_model_input
from model_adapter import predict



def find_patient_with_observations(limit=20):
    patients = get_patients(limit=limit)

    for entry in patients:
        patient = entry.get("resource", {})
        patient_id = patient.get("id")

        if not patient_id:
            continue

        observations = get_observations(patient_id)

        if len(observations) > 0:
            return patient_id, observations

    return None, []


def main():
    print("Fetching patients...")

    patient_id, observations = find_patient_with_observations(limit=20)

    if not patient_id:
        print("No patient with observations found in sampled set.")
        return

    print(f"\nSelected Patient ID: {patient_id}")
    print(f"Observation count pulled: {len(observations)}")

    patient_data = get_patient(patient_id)
    mapped = map_fhir_to_features(patient_data, observations)

    print("\nMapped Features:")
    for k, v in mapped.items():
        if k != "source":
            print(f"{k}: {v}")

    print("\nSource Labels:")
    pprint(mapped["source"])

    model_input = prepare_model_input(mapped)

    print("\nModel Input (Scaled):")
    print(model_input.head())

    pred, prob = predict(mapped)

    print("\nPrediction:", pred)
    print("Risk Probability:", prob)


if __name__ == "__main__":
    main()