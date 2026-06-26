"""
Generates a synthetic borrowing-history CSV used to train the
demand-prediction model (Section B, Q1-c).

Run once:
    python data/generate_borrowing_history.py
"""
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

HERE = Path(__file__).parent
with open(HERE / "catalogue.json") as f:
    catalogue = json.load(f)

start_date = datetime(2024, 1, 1)
rows = []

for book in catalogue:
    # Give each book a "popularity" weight so some books are clearly high-demand
    popularity = random.uniform(0.2, 1.0)
    if book["section"] == "Computer Science":
        popularity += 0.3  # CS books borrowed more in this synthetic dataset

    num_borrows = int(popularity * random.randint(20, 80))
    for _ in range(num_borrows):
        borrow_day = start_date + timedelta(days=random.randint(0, 540))
        loan_days = random.choice([7, 14, 14, 21, 30])
        return_day = borrow_day + timedelta(days=loan_days)
        rows.append({
            "isbn": book["isbn"],
            "title": book["title"],
            "section": book["section"],
            "borrow_date": borrow_day.strftime("%Y-%m-%d"),
            "return_date": return_day.strftime("%Y-%m-%d"),
            "semester": "Spring" if borrow_day.month <= 6 else "Fall",
            "copies_total": book["copies_total"],
        })

rows.sort(key=lambda r: r["borrow_date"])

out_path = HERE / "borrowing_history.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} borrowing records to {out_path}")
