"""Exploration, part 1: size of the data and time.

Usage:
    python src/fraud/explore_time.py
Saves the charts in docs/images/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DAY = 86400    # seconds in a day
HOUR = 3600    # seconds in an hour

# ---------------------------------------------------------------- 1. load
df = pd.read_csv("data/raw/train_transaction.csv")

# ---------------------------------------------------------------- 2. size and frauds
print("rows, columns:", df.shape)
print("frauds:", df["isFraud"].sum())
print(f"fraud rate: {df['isFraud'].mean() * 100:.2f}%")

# ---------------------------------------------------------------- 3. time columns
# TransactionDT = seconds from a hidden starting moment. The first value is 86400 = day 1.
df["day"] = df["TransactionDT"] // DAY            # whole days since the start
df["month"] = df["day"] // 30                     # 30-day blocks
df.loc[df["month"] == 6, "month"] = 5             # "month 6" is only the last 2 days: join it to month 5
df["hour"] = (df["TransactionDT"] // HOUR) % 24   # 0..23 (0 is not necessarily midnight)
df["weekday"] = df["day"] % 7                     # 0..6 (we don't know which one is Monday)
df = df.copy()                                    # removes the "fragmented" warning

print("days covered:", df["day"].max() - df["day"].min())

# ---------------------------------------------------------------- 4. per month
by_month = df.groupby("month").agg(
    purchases=("isFraud", "size"),
    frauds=("isFraud", "sum"),
    fraud_rate=("isFraud", "mean"),
    amount=("TransactionAmt", "sum"),
)
by_month["fraud_rate"] = (by_month["fraud_rate"] * 100).round(2)
print("\nper month:")
print(by_month)


# ---------------------------------------------------------------- 5. charts: per hour and per weekday
def chart(column: str, n_values: int, title: str, file_name: str) -> None:
    """Two bar charts, one above the other: how many purchases, and the % of frauds."""
    grouped = df.groupby(column).agg(
        purchases=("isFraud", "size"),   # how many purchases
        frauds=("isFraud", "mean"),      # share of frauds
    )
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    top.bar(grouped.index, grouped["purchases"], color="grey")
    top.set_ylabel("purchases")
    top.set_title(title)

    bottom.bar(grouped.index, grouped["frauds"] * 100, color="red")
    bottom.set_ylabel("frauds (%)")
    bottom.set_xlabel(column)
    bottom.set_xticks(range(n_values))

    plt.tight_layout()
    Path("docs/images").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"docs/images/{file_name}", dpi=120)
    plt.show()


chart("hour", 24, "Purchases and frauds per hour of the day", "per_hour.png")
chart("weekday", 7, "Purchases and frauds per day of the week", "per_weekday.png")
print("\ncharts -> docs/images/per_hour.png, docs/images/per_weekday.png")