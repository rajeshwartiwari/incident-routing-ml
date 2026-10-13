import csv
import tempfile
import unittest
from pathlib import Path

from src.generate_synthetic_data import (
    FIELDNAMES,
    generate_incidents,
    split_train_test,
    write_incidents_csv,
)


class GenerateIncidentsTests(unittest.TestCase):
    def test_generates_requested_number_with_expected_schema(self):
        incidents = generate_incidents(row_count=25, seed=7)

        self.assertEqual(len(incidents), 25)
        self.assertEqual(tuple(incidents[0]), FIELDNAMES)
        self.assertEqual(len({row["incident_id"] for row in incidents}), 25)
        self.assertTrue(all(row["description"] for row in incidents))

    def test_generation_is_reproducible_for_same_seed(self):
        self.assertEqual(
            generate_incidents(row_count=10, seed=19),
            generate_incidents(row_count=10, seed=19),
        )

    def test_priority_is_consistent_with_urgency_and_impact(self):
        incidents = generate_incidents(row_count=100, seed=3)
        expected = {
            ("High", "High"): "P1",
            ("High", "Medium"): "P2",
            ("Medium", "High"): "P2",
            ("Low", "Low"): "P4",
        }

        for incident in incidents:
            pair = (incident["urgency"], incident["impact"])
            if pair in expected:
                self.assertEqual(incident["priority"], expected[pair])

    def test_rejects_non_positive_row_count(self):
        with self.assertRaises(ValueError):
            generate_incidents(row_count=0)

    def test_split_is_80_20_disjoint_and_reproducible(self):
        incidents = generate_incidents(row_count=100, seed=5)
        train, test = split_train_test(incidents, seed=5)

        self.assertEqual((len(train), len(test)), (80, 20))
        train_ids = {row["incident_id"] for row in train}
        test_ids = {row["incident_id"] for row in test}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(train_ids | test_ids), 100)
        self.assertEqual((train, test), split_train_test(incidents, seed=5))

    def test_time_based_split_ensures_newer_records_in_test(self):
        incidents = generate_incidents(row_count=100, seed=5)
        train, test = split_train_test(incidents, time_based=True)

        max_train_time = max(r["created_at"] for r in train)
        min_test_time = min(r["created_at"] for r in test)
        self.assertLessEqual(max_train_time, min_test_time)

    def test_writes_csv_with_header_and_rows(self):
        incidents = generate_incidents(row_count=3, seed=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "incidents.csv"
            write_incidents_csv(incidents, output)

            with output.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 3)
        self.assertEqual(tuple(rows[0]), FIELDNAMES)


if __name__ == "__main__":
    unittest.main()
