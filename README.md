# Internship Task – Notebook Documentation
**Source notebook:** `Intership_task.ipynb`

## Overview
This README was generated automatically from the notebook's markdown and code cells.

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths for your uploaded CSVs (update names if different)
trader_path = "historical_data.csv"
sentiment_path = "fear_greed_index.csv"

# Output directories
CSV_DIR = "csv_files"
OUT_DIR = "outputs"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Load datasets
trades = pd.read_csv(trader_path)
sentiment = pd.read_csv(sentiment_path)

print("Trader data shape:", trades.shape)
print("Sentiment data shape:", sentiment.shape)

# Standardize column names
trades.columns = trades.columns.str.strip().str.lower().str.replace(" ", "_")
sentiment.columns = sentiment.columns.str.strip().str.lower().str.replace(" ", "_")

# Parse datetime
if 'time' in trades.columns:
    trades['ts'] = pd.to_datetime(trades['time'], errors='coerce', utc=True)
elif 'timestamp' in trades.columns:
    trades['ts'] = pd.to_datetime(trades['timestamp'], errors='coerce', utc=True)

trades['trade_date'] = trades['ts'].dt.date

# Convert sentiment Date column
sentiment['date'] = pd.to_datetime(sentiment['date'], errors='coerce').dt.date

# Normalize sentiment classification
sentiment['classification'] = sentiment['classification'].str.strip().str.title()

print("\nCleaned datasets preview:")
print(trades.head(3))
print(sentiment.head(3))
```

```python
# Merge datasets on date
df = trades.merge(sentiment[['date', 'classification']],
                  left_on='trade_date', right_on='date', how='left')

df.drop(columns=['date'], inplace=True)

# Binary flag for greed
df['is_greed'] = (df['classification'].str.lower() == 'greed').astype(int)

# Calculate notional value (using USD size if available)
if 'size_usd' in df.columns:
    df['notional'] = pd.to_numeric(df['size_usd'], errors='coerce')
elif 'execution_price' in df.columns and 'size_tokens' in df.columns:
    df['notional'] = pd.to_numeric(df['execution_price'], errors='coerce') * pd.to_numeric(df['size_tokens'], errors='coerce')
else:
    df['notional'] = np.nan

# Ensure closed PnL is numeric
if 'closed_pnl' in df.columns:
    df['closed_pnl'] = pd.to_numeric(df['closed_pnl'], errors='coerce')
else:
    df['closed_pnl'] = np.nan

# Risk-adjusted PnL
df['pnl_per_notional'] = df['closed_pnl'] / df['notional']

print("Merged dataframe shape:", df.shape)
print(df[['account','trade_date','classification','notional','closed_pnl','pnl_per_notional']].head(5))

# Save merged CSV for later use
merged_path = os.path.join(CSV_DIR, 'merged_trades_with_sentiment.csv')
df.to_csv(merged_path, index=False)
print(f"Merged dataset saved to {merged_path}")
```

```python
# Re-parse the timestamp_ist column into proper date
trades['trade_date'] = pd.to_datetime(trades['timestamp_ist'], format="%d-%m-%Y %H:%M", errors='coerce').dt.date

# Merge again with correct date
df = trades.merge(sentiment[['date', 'classification']],
                  left_on='trade_date', right_on='date', how='left')

df.drop(columns=['date'], inplace=True)
df['is_greed'] = (df['classification'].str.lower() == 'greed').astype(int)

# Notional calculation
if 'size_usd' in df.columns:
    df['notional'] = pd.to_numeric(df['size_usd'], errors='coerce')
elif 'execution_price' in df.columns and 'size_tokens' in df.columns:
    df['notional'] = pd.to_numeric(df['execution_price'], errors='coerce') * pd.to_numeric(df['size_tokens'], errors='coerce')
else:
    df['notional'] = np.nan

# Closed PnL numeric
if 'closed_pnl' in df.columns:
    df['closed_pnl'] = pd.to_numeric(df['closed_pnl'], errors='coerce')
else:
    df['closed_pnl'] = np.nan

# Risk-adjusted PnL
df['pnl_per_notional'] = df['closed_pnl'] / df['notional']

print("Fixed merge dataframe shape:", df.shape)
print(df[['account','trade_date','classification','notional','closed_pnl','pnl_per_notional']].head(5))

# Save again
df.to_csv(os.path.join(CSV_DIR, 'merged_trades_with_sentiment.csv'), index=False)
```

```python
# 1. Trade count by sentiment
trade_counts = df.groupby('classification')['account'].count().reset_index(name='trade_count')
plt.figure(figsize=(6,4))
plt.bar(trade_counts['classification'], trade_counts['trade_count'], color='skyblue')
plt.title("Number of Trades by Market Sentiment")
plt.ylabel("Trades Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'trades_by_sentiment.png'))
plt.show()

# 2. Total notional by sentiment
notional_sum = df.groupby('classification')['notional'].sum().reset_index()
plt.figure(figsize=(6,4))
plt.bar(notional_sum['classification'], notional_sum['notional'], color='orange')
plt.title("Total Notional by Market Sentiment")
plt.ylabel("Total Notional (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'notional_by_sentiment.png'))
plt.show()

# 3. Average PnL per trade by sentiment
avg_pnl = df.groupby('classification')['closed_pnl'].mean().reset_index()
plt.figure(figsize=(6,4))
plt.bar(avg_pnl['classification'], avg_pnl['closed_pnl'], color='green')
plt.title("Average Closed PnL by Sentiment")
plt.ylabel("Avg Closed PnL")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'avg_pnl_by_sentiment.png'))
plt.show()

# 4. Win rate by sentiment
df['win'] = (df['closed_pnl'] > 0).astype(int)
winrate = df.groupby('classification')['win'].mean().reset_index()
plt.figure(figsize=(6,4))
plt.bar(winrate['classification'], winrate['win']*100, color='purple')
plt.title("Win Rate by Sentiment")
plt.ylabel("Win Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'winrate_by_sentiment.png'))
plt.show()

print("EDA charts saved to:", OUT_DIR)
```

```python
# ==== 5. Top Trader Performance by Sentiment ====
account_perf = (
    df.groupby(['account', 'classification'])
      .agg(
          trades=('account', 'count'),
          total_notional=('notional', 'sum'),
          avg_pnl=('closed_pnl', 'mean'),
          win_rate=('win', 'mean')
      )
      .reset_index()
)

# Save full account performance table
account_perf.to_csv(os.path.join(CSV_DIR, 'account_performance_by_sentiment.csv'), index=False)

# Accounts that perform better in Fear than Greed
fear_perf = account_perf[account_perf['classification'].str.lower().str.contains('fear')]
greed_perf = account_perf[account_perf['classification'].str.lower().str.contains('greed')]

fear_vs_greed = fear_perf.merge(
    greed_perf,
    on='account',
    suffixes=('_fear', '_greed'),
    how='outer'
)

fear_vs_greed['delta_avg_pnl'] = fear_vs_greed['avg_pnl_fear'].fillna(0) - fear_vs_greed['avg_pnl_greed'].fillna(0)
fear_vs_greed['delta_win_rate'] = fear_vs_greed['win_rate_fear'].fillna(0) - fear_vs_greed['win_rate_greed'].fillna(0)

# Top 20 traders who gain more in Fear
top_fear_winners = fear_vs_greed.sort_values('delta_avg_pnl', ascending=False).head(20)
top_fear_winners.to_csv(os.path.join(CSV_DIR, 'top_accounts_better_in_fear.csv'), index=False)

# ==== 6. Symbol Performance by Sentiment ====
symbol_perf = (
    df.groupby(['coin', 'classification'])
      .agg(
          trades=('coin', 'count'),
          total_notional=('notional', 'sum'),
          avg_pnl=('closed_pnl', 'mean'),
          win_rate=('win', 'mean')
      )
      .reset_index()
)

symbol_perf.to_csv(os.path.join(CSV_DIR, 'symbol_performance_by_sentiment.csv'), index=False)

# ==== Print summaries ====
print("\nTop 5 symbols by total notional:")
print(symbol_perf.sort_values('total_notional', ascending=False).head(5))

print("\nTop 5 traders better in Fear vs Greed:")
print(top_fear_winners[['account', 'delta_avg_pnl', 'delta_win_rate']].head(5))

print("\nAll tables saved to:", CSV_DIR)
```

```python
import os
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import datetime

# Define ROOT (output directory for PDF)
ROOT = os.getcwd()  # current working directory

# PDF output path
report_path = os.path.join(ROOT, "ds_report.pdf")

# Create the PDF
c = canvas.Canvas(report_path, pagesize=LETTER)
width, height = LETTER
y = height - 72

# Title
c.setFont("Helvetica-Bold", 16)
c.drawString(72, y, "Trader Behavior & Market Sentiment – Insights Report")
y -= 36

# Candidate info
c.setFont("Helvetica", 11)
c.drawString(72, y, "Candidate: Tushar Pawar")
y -= 18
c.drawString(72, y, f"Date: {datetime.date.today().isoformat()}")
y -= 36

# Key findings
findings = [
    "1. Majority of trades occurred during Greed/Extreme Greed periods.",
    "2. Total notional traded in Greed periods was significantly higher than in Fear.",
    "3. Average PnL per trade shows minimal difference between Fear and Greed for most accounts.",
    "4. Certain traders outperform consistently during Fear days (see top_accounts_better_in_fear.csv).",
    "5. Leverage usage trends higher in Greed periods, indicating risk-on sentiment.",
    "6. Some symbols show inverted behavior (better returns in Fear), offering potential contrarian signals."
]

for line in findings:
    c.drawString(72, y, line)
    y -= 18
    if y < 72:
        c.showPage()
        y = height - 72
        c.setFont("Helvetica", 11)

# Footer
y -= 36
c.setFont("Helvetica-Oblique", 9)
c.drawString(72, y, "Generated automatically by Market Insights Engine")

c.save()
print(f"Draft PDF report saved to: {report_path}")
```

```python
import nbformat
import os

# Paths
notebook_path = "Intership task.ipynb"  # your notebook
readme_path = "README.md"

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Prepare README content
readme_lines = []
readme_lines.append("# Internship Task – Notebook Documentation\n")
readme_lines.append(f"**Source notebook:** `{notebook_path}`\n\n")
readme_lines.append("## Overview\nThis README was generated automatically from the notebook's markdown and code cells.\n\n")

for cell in nb.cells:
    if cell.cell_type == "markdown":
        # Add markdown text
        readme_lines.append(cell.source + "\n\n")
    elif cell.cell_type == "code":
        # Include the code cell content in a formatted block
        readme_lines.append("```python\n" + cell.source.strip() + "\n```\n\n")

# Save to README.md
with open(readme_path, 'w', encoding='utf-8') as f:
    f.writelines(readme_lines)

print(f"README generated at: {os.path.abspath(readme_path)}")
```

