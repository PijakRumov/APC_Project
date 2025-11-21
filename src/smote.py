import pandas as pd
from imblearn.over_sampling import SMOTE
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'output_train.csv')

df = pd.read_csv(csv_path)

X = df.drop(columns=['Label'])
Y = df['Label']

print('SMOTE started ...')

smote = SMOTE(
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42,
)
X_res, Y_res = smote.fit_resample(X, Y)

df_res = X_res.copy()
df_res['Label'] = Y_res
df_res.to_csv(os.path.join(script_dir, 'output_train_smote.csv'), index=False)

print('SMOTE applied and balanced dataset saved as output_train_smote.csv')
