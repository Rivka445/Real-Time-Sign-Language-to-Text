import pandas as pd
import numpy as np

print("🔄 קורא את הדאטהסט הקיים...")
df = pd.read_csv('dataset.csv')

# שומרים את עמודת הלייבל בצד
labels = df['label']
# מורידים את עמודת הלייבל כדי לעבוד רק על המספרים
features = df.drop('label', axis=1).values

print("🧮 מנרמל את נקודות הציון יחסית לשורש כף היד (נקודה 0)...")
# נעבור שורה שורה ונתקן אותה
for row_idx in range(len(features)):
    # יד ראשונה (Hand 0) - נקודות 0 עד 62
    h0_base_x = features[row_idx, 0]
    h0_base_y = features[row_idx, 1]
    h0_base_z = features[row_idx, 2]
    
    # אם היד הזו לא ריקה מאפסים, נחסיר את נקודת הבסיס שלה
    if not (h0_base_x == 0 and h0_base_y == 0 and h0_base_z == 0):
        for i in range(21):
            features[row_idx, i*3]   -= h0_base_x
            features[row_idx, i*3+1] -= h0_base_y
            features[row_idx, i*3+2] -= h0_base_z

    # יד שנייה (Hand 1) - נקודות 63 עד 125
    h1_base_x = features[row_idx, 63]
    h1_base_y = features[row_idx, 64]
    h1_base_z = features[row_idx, 65]
    
    if not (h1_base_x == 0 and h1_base_y == 0 and h1_base_z == 0):
        for i in range(21):
            features[row_idx, 63 + i*3]   -= h1_base_x
            features[row_idx, 63 + i*3+1] -= h1_base_y
            features[row_idx, 63 + i*3+2] -= h1_base_z

# בניית ה-DataFrame המעודכן ושמירתו
normalized_df = pd.DataFrame(features, columns=df.columns[1:])
normalized_df.insert(0, 'label', labels)

normalized_df.to_csv('dataset.csv', index=False)
