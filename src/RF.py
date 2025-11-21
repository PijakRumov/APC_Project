import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
print("Training Random Forest Classifier...")

df_train = pd.read_csv(os.path.join(script_dir, 'output_train.csv'))
df_test = pd.read_csv(os.path.join(script_dir, 'output_test.csv'))

X_train = df_train.drop(columns=['Label'])
Y_train = df_train['Label'].values

model = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, Y_train)
print("Random Forest model trained successfully.")

X_test = df_test
Y_pred = model.predict(X_test)

df_pred = pd.DataFrame(Y_pred, columns=['Predicted_Label'])
df_pred.to_csv('predictions_test.csv', index=False)
print("Predictions saved")


