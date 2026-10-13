# src/test_bad_data.py
"""Test what happens when we send garbage data to the incident routing API."""
import requests

BASE_URL = "http://localhost:8001"

GOOD = {
    "description": "User cannot access mailbox after password reset",
    "urgency": "High",
    "impact": "Medium",
}

# (title, payload) - each payload starts from a valid incident and breaks it
CASES = [
    ("Invalid urgency value", {**GOOD, "urgency": "Critical"}),
    ("Invalid impact value", {**GOOD, "impact": "Huge"}),
    ("Empty description", {**GOOD, "description": ""}),
    ("Whitespace-only description", {**GOOD, "description": "     "}),
    ("Missing description field", {"urgency": "High", "impact": "Medium"}),
    ("Wrong type for description", {**GOOD, "description": 12345}),
    ("Gibberish description (no known words)", {**GOOD, "description": "asdf qwer zxcv"}),
    ("Very long description (100,000 chars)", {**GOOD, "description": "mailbox " * 12500}),
    (
        "Everything wrong",
        {"description": "", "urgency": "ASAP", "impact": None},
    ),
]


def main():
    print("Testing API with various bad inputs...\n")
    accepted = []

    for i, (title, payload) in enumerate(CASES, start=1):
        print(f"Test {i}: {title}")
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        print(f"  Status: {response.status_code}")
        body = response.json()
        if response.status_code == 200:
            accepted.append(title)
            print(f"  Prediction: {body['assignment_group']} (confidence {body['confidence']})\n")
        else:
            print(f"  Response: {body}\n")

    print(f"{len(accepted)} of {len(CASES)} bad inputs were accepted and given a prediction.")
    if accepted:
        print("Accepted despite being bad:")
        for title in accepted:
            print(f"  - {title}")
        print("The API returns a routing decision with no warning, e.g. for text it cannot understand.")
    print("Only the fields with a fixed schema (urgency, impact) are strictly validated.")


if __name__ == "__main__":
    main()
