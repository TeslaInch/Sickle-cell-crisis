import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n = 500

data = {
    "pain_level": np.random.randint(1, 11, n),
    "hydration": np.round(np.random.uniform(0.5, 5.0, n), 1),
    "medication": np.random.randint(0, 2, n),
    "temperature": np.round(np.random.uniform(36.0, 40.0, n), 1),
    "fatigue": np.random.randint(1, 11, n)
}

df = pd.DataFrame(data)

# Define a simple rule-based crisis probability
prob = (
    (df["pain_level"] / 10) +
    (1 - (df["hydration"] / 5)) +
    (df["fatigue"] / 10) +
    ((df["temperature"] - 36) / 4)
) / 4

df["crisis"] = np.where(prob > 0.5, 1, 0)

# Save dataset
df.to_csv("data/sickle_cell_data.csv", index=False)
print("Dataset saved to data/sickle_cell_data.csv")
print(df.head())
