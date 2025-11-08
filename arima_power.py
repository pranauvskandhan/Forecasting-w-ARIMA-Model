import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# dataset
ROOT = Path(__file__).resolve().parent
PATH = ROOT / "datasets" / "AEP_hourly.csv"

# ------------------------ 1. Load and preprocess data ------------------------
df = pd.read_csv(PATH)
df.columns = [c.strip() for c in df.columns]
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime').sort_index()

# location for plots
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

#  there are duplictae timestamps 

#fix 
df = df.groupby(level=0).mean()

# MW column
mw_col = [c for c in df.columns if c.endswith('_MW')][0]

# make hourly continuous index & fill small gaps
all_hours = pd.date_range(df.index.min(), df.index.max(), freq='H')
df = df.reindex(all_hours)
df[mw_col] = df[mw_col].interpolate(limit_direction='both')

# verify visual 
plt.figure(figsize=(12,4))
plt.plot(df.index, df[mw_col], linewidth=1)
plt.title("AEP Hourly Electricity Demand (MW)")
plt.xlabel("Time"); plt.ylabel("MW")
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_raw_timeseries.png", dpi=300)
plt.close()

# display
img = mpimg.imread(PLOT_DIR / "01_raw_timeseries.png")
plt.figure(figsize=(12,4))
plt.imshow(img)
plt.axis('off')
plt.show()


# ------------------------ 2. Train-Test Split ------------------------
#(last 30 days = 720 hours for testing)

h_test = 24*30
train = df.iloc[:-h_test].copy()
test = df.iloc[-h_test:].copy()

y_train = train[mw_col]
y_test = test[mw_col]

print(f"Train: {y_train.index.min()} → {y_train.index.max()}  (n={len(y_train)})")
print(f"Test : {y_test.index.min()} → {y_test.index.max()}   (n={len(y_test)})")

# Train: 2004-10-01 01:00:00 → 2018-07-04 00:00:00  (n=120576)
# Test : 2018-07-04 01:00:00 → 2018-08-03 00:00:00   (n=720)

# stop