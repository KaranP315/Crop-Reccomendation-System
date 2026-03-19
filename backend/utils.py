"""
utils.py — Helper utilities for the Crop Recommendation backend
================================================================
Contains input-validation logic so that app.py stays clean.
"""

# The 7 features the model expects
REQUIRED_FIELDS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Reasonable value ranges for each feature (used for soft validation warnings)
FIELD_RANGES = {
    "N":           (0, 140),
    "P":           (5, 145),
    "K":           (5, 205),
    "temperature": (0, 50),
    "humidity":    (10, 100),
    "ph":          (0, 14),
    "rainfall":    (20, 300),
}


def validate_input(data: dict) -> tuple[bool, str]:
    """
    Validate that *data* contains all required fields and that every
    value is numeric.

    Returns
    -------
    (is_valid, error_message)
        is_valid is True when the data is acceptable, False otherwise.
        error_message is "" on success, or a human-readable explanation on failure.
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object."

    # Check for missing fields
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"

    # Check that every value is numeric
    for field in REQUIRED_FIELDS:
        value = data[field]
        if not isinstance(value, (int, float)):
            return False, f"Field '{field}' must be a number, got {type(value).__name__}."

    return True, ""
