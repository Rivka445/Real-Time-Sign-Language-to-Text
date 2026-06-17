import pandas as pd
import numpy as np

print("🔄 Loading the existing dataset...")
df = pd.read_csv('dataset.csv')

# Separate the label column from the coordinates
labels = df['label']
features = df.drop('label', axis=1).values

print("🧮 Normalizing landmarks relative to the wrist (Point 0)...")
# Iterate row by row to normalize the coordinates
for row_idx in range(len(features)):
    # First Hand (Hand 0) - Points 0 to 62 in the feature array
    h0_base_x = features[row_idx, 0]
    h0_base_y = features[row_idx, 1]
    h0_base_z = features[row_idx, 2]
    
    # If Hand 0 is detected (not padded with zeros), subtract the base point
    if not (h0_base_x == 0 and h0_base_y == 0 and h0_base_z == 0):
        for i in range(21):
            features[row_idx, i*3]   -= h0_base_x
            features[row_idx, i*3+1] -= h0_base_y
            features[row_idx, i*3+2] -= h0_base_z

    # Second Hand (Hand 1) - Points 63 to 125 in the feature array
    h1_base_x = features[row_idx, 63]
    h1_base_y = features[row_idx, 64]
    h1_base_z = features[row_idx, 65]
    
    # If Hand 1 is detected (not padded with zeros), subtract the base point
    if not (h1_base_x == 0 and h1_base_y == 0 and h1_base_z == 0):
        for i in range(21):
            features[row_idx, 63 + i*3]   -= h1_base_x
            features[row_idx, 63 + i*3+1] -= h1_base_y
            features[row_idx, 63 + i*3+2] -= h1_base_z

# Reconstruct the updated DataFrame and save it back to CSV
normalized_df = pd.DataFrame(features, columns=df.columns[1:])
normalized_df.insert(0, 'label', labels)

normalized_df.to_csv('dataset.csv', index=False)
print("✅ Dataset successfully normalized! Ready for retraining.")