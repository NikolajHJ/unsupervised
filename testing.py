import pandas as pd
# Adjust the file name if yours differs
raw_path = "heart_attack_prediction_indonesia.csv"

df = pd.read_csv(raw_path, low_memory=False)

# Display the distribution of the raw (unaltered) 'alcohol_consumption' column
dist = df["alcohol_consumption"].value_counts(dropna=False).rename_axis("alcohol_consumption").reset_index(name="count")
print(dist)