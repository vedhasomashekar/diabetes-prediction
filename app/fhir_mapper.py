from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


GLUCOSE_CODES = {
    "2339-0",   
    "2345-7",   
    "41653-7",  

BP_PANEL_CODE = "85354-9"
SYSTOLIC_CODE = "8480-6"
DIASTOLIC_CODE = "8462-4"
WEIGHT_CODE = "29463-7"
HEIGHT_CODE = "8302-2"
BMI_CODE = "39156-5"


def _safe_parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def extract_age(patient_json: Dict[str, Any]) -> Optional[int]:
    birth_date = patient_json.get("birthDate")
    if not birth_date:
        return None

    try:
        dob = datetime.strptime(birth_date, "%Y-%m-%d").date()
    except ValueError:
        try:
            dob = datetime.strptime(birth_date, "%Y-%m").date()
        except ValueError:
            try:
                dob = datetime.strptime(birth_date, "%Y").date()
            except ValueError:
                return None

    today = datetime.today().date()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age


def _get_value_quantity(obs: Dict[str, Any]) -> Optional[float]:
    value = obs.get("valueQuantity", {}).get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_observation_codes(obs: Dict[str, Any]) -> List[str]:
    codings = obs.get("code", {}).get("coding", [])
    return [str(c.get("code", "")).strip() for c in codings if c.get("code")]


def _get_effective_datetime(obs: Dict[str, Any]) -> Optional[datetime]:
    return (
        _safe_parse_datetime(obs.get("effectiveDateTime"))
        or _safe_parse_datetime(obs.get("issued"))
    )


def _choose_latest(candidate: Tuple[Optional[datetime], Optional[float]],
                   current: Tuple[Optional[datetime], Optional[float]]) -> Tuple[Optional[datetime], Optional[float]]:
    cand_dt, cand_val = candidate
    curr_dt, curr_val = current

    if cand_val is None:
        return current
    if curr_val is None:
        return candidate
    if cand_dt and curr_dt:
        return candidate if cand_dt > curr_dt else current
    if cand_dt and not curr_dt:
        return candidate
    return current


def extract_observation_values(observations: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    latest = {
        "glucose": (None, None),
        "systolic_bp": (None, None),
        "diastolic_bp": (None, None),
        "weight": (None, None),
        "height": (None, None),
        "bmi_direct": (None, None),
    }

    for entry in observations:
        obs = entry.get("resource", {})
        obs_dt = _get_effective_datetime(obs)
        codes = set(_get_observation_codes(obs))

        if codes & GLUCOSE_CODES:
            latest["glucose"] = _choose_latest((obs_dt, _get_value_quantity(obs)), latest["glucose"])

        if WEIGHT_CODE in codes:
            latest["weight"] = _choose_latest((obs_dt, _get_value_quantity(obs)), latest["weight"])

        if HEIGHT_CODE in codes:
            latest["height"] = _choose_latest((obs_dt, _get_value_quantity(obs)), latest["height"])

        if BMI_CODE in codes:
            latest["bmi_direct"] = _choose_latest((obs_dt, _get_value_quantity(obs)), latest["bmi_direct"])

        if BP_PANEL_CODE in codes:
            sys_val = None
            dia_val = None

            for comp in obs.get("component", []):
                comp_codes = {
                    str(c.get("code", "")).strip()
                    for c in comp.get("code", {}).get("coding", [])
                    if c.get("code")
                }
                comp_val = comp.get("valueQuantity", {}).get("value")
                try:
                    comp_val = float(comp_val) if comp_val is not None else None
                except (TypeError, ValueError):
                    comp_val = None

                if SYSTOLIC_CODE in comp_codes:
                    sys_val = comp_val
                if DIASTOLIC_CODE in comp_codes:
                    dia_val = comp_val

            latest["systolic_bp"] = _choose_latest((obs_dt, sys_val), latest["systolic_bp"])
            latest["diastolic_bp"] = _choose_latest((obs_dt, dia_val), latest["diastolic_bp"])

    return {
        "glucose": latest["glucose"][1],
        "systolic_bp": latest["systolic_bp"][1],
        "diastolic_bp": latest["diastolic_bp"][1],
        "weight": latest["weight"][1],
        "height": latest["height"][1],
        "bmi_direct": latest["bmi_direct"][1],
    }


def compute_bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    if weight_kg is None or height_cm is None or height_cm == 0:
        return None
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def map_fhir_to_features(patient_json: Dict[str, Any], observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    age = extract_age(patient_json)
    obs_values = extract_observation_values(observations)

    if obs_values["bmi_direct"] is not None:
        bmi = obs_values["bmi_direct"]
        bmi_source = "FHIR"
    else:
        bmi = compute_bmi(obs_values["weight"], obs_values["height"])
        bmi_source = "derived" if bmi is not None else "missing"

    mapped = {
        "patient_id": patient_json.get("id"),
        "age": age,
        "glucose": obs_values["glucose"],
        "systolic_bp": obs_values["systolic_bp"],
        "diastolic_bp": obs_values["diastolic_bp"],
        "bmi": bmi,
        "raw_height": obs_values["height"],
        "raw_weight": obs_values["weight"],
        "source": {
            "age": "FHIR" if age is not None else "missing",
            "glucose": "FHIR" if obs_values["glucose"] is not None else "missing",
            "systolic_bp": "FHIR" if obs_values["systolic_bp"] is not None else "missing",
            "diastolic_bp": "FHIR" if obs_values["diastolic_bp"] is not None else "missing",
            "bmi": bmi_source,
            "raw_height": "FHIR" if obs_values["height"] is not None else "missing",
            "raw_weight": "FHIR" if obs_values["weight"] is not None else "missing",
        }
    }

    return mapped