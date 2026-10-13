"""Generate fictional IT incidents for the incident-routing project."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


FIELDNAMES = (
    "incident_id",
    "created_at",
    "description",
    "category",
    "subcategory",
    "urgency",
    "impact",
    "priority",
    "assignment_group",
)

_PROFILES = (
    (
        "Hardware",
        "Laptop",
        "End User Computing",
        (
            "Laptop {symptom} after {trigger}",
            "{symptom} on the company laptop prevents {work}",
        ),
        ("slow", "won't start", "overheats", "keeps disconnecting"),
        ("a restart", "the latest update", "returning from sleep"),
        ("joining video calls", "accessing work files", "working remotely"),
    ),
    (
        "Hardware",
        "Printer",
        "End User Computing",
        (
            "Office printer {symptom} when {trigger}",
            "Unable to print because the shared printer {symptom}",
        ),
        ("is offline", "prints blank pages", "shows a paper jam", "rejects jobs"),
        ("sending a print job", "changing paper", "reconnecting to Wi-Fi"),
        ("printing a report", "printing shipping labels", "scanning a document"),
    ),
    (
        "Software",
        "Business application",
        "Application Support",
        (
            "{system} {symptom} while {trigger}",
            "Users cannot complete {work} in {system}; it {symptom}",
        ),
        ("shows an error", "loads slowly", "closes unexpectedly", "does not save changes"),
        ("opening a record", "submitting the form", "the latest release"),
        ("processing an order", "updating a customer record", "closing the daily report"),
    ),
    (
        "Software",
        "Operating system",
        "End User Computing",
        (
            "Workstation {symptom} after {trigger}",
            "{system} {symptom} while the user is {trigger}",
        ),
        ("fails to install updates", "restarts unexpectedly", "cannot detect the display"),
        ("a system update", "connecting a docking station", "signing in"),
        ("starting the workday", "sharing a presentation", "using an internal application"),
    ),
    (
        "Network",
        "Connectivity",
        "Network Operations",
        (
            "{system} {symptom} when {trigger}",
            "Intermittent network connection is affecting {work}",
        ),
        ("is unreachable", "drops the connection", "has high latency", "cannot resolve the host"),
        ("connecting through VPN", "switching networks", "the office connection was restored"),
        ("accessing internal tools", "joining a video call", "reaching a shared drive"),
    ),
    (
        "Access",
        "Account and authentication",
        "Identity and Access Management",
        (
            "User {symptom} when {trigger}",
            "{system} access is blocked; the user {symptom}",
        ),
        ("cannot sign in", "is repeatedly prompted for MFA", "is locked out", "has lost access"),
        ("resetting a password", "enrolling a new phone", "returning from leave"),
        ("signing in to the portal", "approving an MFA prompt", "accessing a team workspace"),
    ),
    (
        "Email",
        "Mailbox",
        "Messaging",
        (
            "{system} {symptom} after {trigger}",
            "Mailbox issue prevents the user from {work}",
        ),
        ("does not send messages", "is missing new messages", "cannot open attachments"),
        ("moving to a new device", "the mailbox migration", "the latest client update"),
        ("sending customer updates", "searching for an older message", "sharing a calendar"),
    ),
    (
        "Database",
        "Performance and availability",
        "Database Operations",
        (
            "{system} {symptom} while {trigger}",
            "Application reports database errors when {work}",
        ),
        ("connections are timing out", "queries run slowly", "is unavailable", "returns stale results"),
        ("the scheduled maintenance", "a reporting query runs", "the morning data refresh"),
        ("loading the analytics dashboard", "processing transactions", "running the monthly report"),
    ),
    (
        "Security",
        "Suspicious activity",
        "Security Operations",
        (
            "Possible phishing message reported after {trigger}",
            "User noticed {symptom} in {system}",
        ),
        ("an unexpected sign-in prompt", "a suspicious attachment", "an unfamiliar login alert"),
        ("opening an unexpected email", "receiving an MFA request", "reviewing account activity"),
        ("verifying a password reset", "opening an attachment", "checking a sign-in notification"),
    ),
    (
        "General",
        "Service request",
        "Service Desk",
        (
            "Request for help with {system} while {trigger}",
            "User needs assistance {work}",
        ),
        ("needs a configuration change", "is not clear to the user", "requires a new setup"),
        ("setting up a new workstation", "starting a new project", "preparing for a team change"),
        ("requesting access to a shared resource", "setting up a workstation", "finding the right support team"),
    ),
)

_SYSTEMS = {
    "Hardware": ("company laptop", "office printer", "workstation"),
    "Software": ("Finance Portal", "Customer Hub", "Windows workstation"),
    "Network": ("office Wi-Fi", "corporate VPN", "internal DNS"),
    "Access": ("identity portal", "MFA account", "company account"),
    "Email": ("Outlook", "shared mailbox", "company calendar"),
    "Database": ("reporting database", "customer database", "analytics warehouse"),
    "Security": ("company account", "email inbox", "identity portal"),
    "General": ("service catalog", "shared workspace", "new workstation"),
}


def _priority(urgency: str, impact: str) -> str:
    """Map urgency and impact to a simple, consistent priority label."""
    if urgency == "High" and impact == "High":
        return "P1"
    if urgency == "High" or impact == "High":
        return "P2"
    if urgency == "Low" and impact == "Low":
        return "P4"
    return "P3"


def generate_incidents(row_count: int = 1000, seed: int = 42) -> List[Dict[str, str]]:
    """Return reproducible fictional incidents with a consistent CSV schema."""
    if row_count < 1:
        raise ValueError("row_count must be at least 1")

    rng = random.Random(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    incidents: List[Dict[str, str]] = []

    for index in range(1, row_count + 1):
        category, subcategory, group, templates, symptoms, triggers, work_items = rng.choice(
            _PROFILES
        )
        urgency = rng.choices(("Low", "Medium", "High"), weights=(2, 5, 3), k=1)[0]
        impact = rng.choices(("Low", "Medium", "High"), weights=(3, 5, 2), k=1)[0]
        system = rng.choice(_SYSTEMS[category])
        template = rng.choice(templates)
        description = template.format(
            system=system,
            symptom=rng.choice(symptoms),
            trigger=rng.choice(triggers),
            work=rng.choice(work_items),
        )
        created_at = start + timedelta(
            minutes=rng.randrange(365 * 24 * 60),
        )

        incidents.append(
            {
                "incident_id": "INC-{0:06d}".format(index),
                "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "description": description,
                "category": category,
                "subcategory": subcategory,
                "urgency": urgency,
                "impact": impact,
                "priority": _priority(urgency, impact),
                "assignment_group": group,
            }
        )

    return incidents


def write_incidents_csv(
    incidents: Sequence[Dict[str, str]], output_path: Path
) -> None:
    """Write incidents to CSV, creating the destination directory if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(incidents)


def split_train_test(
    incidents: Sequence[Dict[str, str]],
    train_ratio: float = 0.8,
    time_based: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split into train and test.
    
    If time_based is True, sorts chronologically by `created_at` so newer
    incidents are reserved for testing (preventing temporal lookahead leakage).
    Otherwise, shuffles randomly using the provided seed.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    if time_based:
        sorted_incidents = sorted(incidents, key=lambda x: x["created_at"])
        train_size = int(len(sorted_incidents) * train_ratio)
        return sorted_incidents[:train_size], sorted_incidents[train_size:]

    shuffled = list(incidents)
    random.Random(seed).shuffle(shuffled)
    train_size = int(len(shuffled) * train_ratio)
    return shuffled[:train_size], shuffled[train_size:]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=_positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for train.csv and test.csv (default: data/raw)",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use random shuffle split instead of default chronological time-based split",
    )
    args = parser.parse_args()

    print("Generating synthetic incident routing dataset...")
    incidents = generate_incidents(row_count=args.rows, seed=args.seed)

    # Split into train (80%) and test (20%) using time-based split by default
    train, test = split_train_test(
        incidents, train_ratio=0.8, time_based=not args.random_split, seed=args.seed
    )

    # Save to CSV files
    write_incidents_csv(train, args.output_dir / "train.csv")
    write_incidents_csv(test, args.output_dir / "test.csv")

    split_mode = "random" if args.random_split else "chronological (time-based)"
    print(f"\nDataset generated successfully ({split_mode} split)!")
    print("Training set: {0:,} incidents".format(len(train)))
    print("Test set: {0:,} incidents".format(len(test)))
    print("\nAssignment group distribution (train):")
    for group, count in sorted(Counter(r["assignment_group"] for r in train).items()):
        print("  {0}: {1:.1%}".format(group, count / len(train)))


if __name__ == "__main__":
    main()
