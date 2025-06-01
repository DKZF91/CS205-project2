import pandas as pd
import numpy as np
from project2_part1 import forward_selection

df = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv")

# Remove spaces from column names and label values
df.columns = [col.strip() for col in df.columns]
df['Classes'] = df['Classes'].str.strip().map({'not fire': 0, 'fire': 1})

# Convert all feature columns to float (except Classes)
feature_columns = [col for col in df.columns if col != 'Classes']
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')

# Remove rows with missing values
df.dropna(inplace=True)

# Splitting features and labels
features = df[feature_columns].to_numpy(dtype=np.float32)
label = df['Classes'].to_numpy(dtype=np.int32)
# Get a list of feature column names (order guaranteed)
feature_names = df[feature_columns].columns.tolist()


print("\nBeginning Forward Selection search...")
best_features, best_acc, history = forward_selection(features, label)

for feature_indices, acc in history:
    selected_names = [feature_names[i] for i in feature_indices]
    print(f"Using feature(s) {selected_names} accuracy is {acc * 100:.1f}%")

best_feature_names = [feature_names[i] for i in best_features]
print(
    f"\nFinished search!! The best feature subset is {best_feature_names}, which has an accuracy of {best_acc * 100:.1f}%.")