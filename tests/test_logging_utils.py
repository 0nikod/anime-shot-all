from pathlib import Path
import csv
import re

from anime_shot_all.logging_utils import write_csv


def test_write_csv_adds_saved_at_timestamp(tmp_path: Path):
    log_path = tmp_path / "logs" / "sample.csv"

    write_csv(log_path, ["name", "value"], [{"name": "alpha", "value": 1}, {"name": "beta", "value": 2}])

    with log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert log_path.exists()
    assert [row["name"] for row in rows] == ["alpha", "beta"]
    assert all(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\+00:00)?$", row["saved_at"]) for row in rows)
    assert len({row["saved_at"] for row in rows}) == 1
