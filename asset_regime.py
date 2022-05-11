# Import standard libraries
from pathlib import Path

# Import numerical computing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure file paths
raw_path = Path('data/raw')
processed_path = Path('data/processed')
results_path = Path('results')

# Display DataFrames without truncation
pd.set_option("display.max_columns", None, "precision", 2)

df = pd.read_csv(raw_path.joinpath("asset_regime.csv"), index_col="Category")

# Add jitter to positions
# rand = np.random.randint(-30, 30, size=df.shape) * 0.01
# df = df + rand

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = df['Growth']
y = df['Inflation']
z = df['Rates']

ax.scatter(x, y, z)
for xi, yi, zi, label  in zip(x, y, z, df.index):
    ax.text(xi, yi, zi, label)

ax.set_xlabel("Growth")
ax.set_ylabel("Inflation")
ax.set_zlabel("Rates")

plt.show()
