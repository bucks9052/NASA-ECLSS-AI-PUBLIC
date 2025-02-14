import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the log data
logs_df = pd.read_csv('eclss_logs.csv')

# Preprocessing
label_encoder = LabelEncoder()
logs_df['AlarmLevel_encoded'] = label_encoder.fit_transform(logs_df['AlarmLevel'])
logs_df['Subsystem_encoded'] = label_encoder.fit_transform(logs_df['Subsystem'])

# Features and target for Decision Tree Classifier
X = logs_df[['Subsystem_encoded', 'SensorValue']]
y = logs_df['AlarmLevel_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Accuracy of the classifier
accuracy = clf.score(X_test, y_test)
print(f"Decision Tree Classifier Accuracy: {accuracy * 100:.2f}%")

# Improved Decision Tree Visualization
plt.figure(figsize=(16,10))
tree.plot_tree(clf, 
               feature_names=['Subsystem', 'SensorValue'], 
               class_names=label_encoder.classes_, 
               filled=True,
               fontsize=10,
               rounded=True,
               proportion=True)
plt.title('Decision Tree for Alarm Classification', fontsize=16)
plt.show()

# Supervised Anomaly Detection (Emergency Classification)
# Create binary labels where Emergency (class 3) is considered anomaly
logs_df['is_emergency'] = (logs_df['AlarmLevel'] == 'Emergency').astype(int)

# Features and target for anomaly detection
X_anomaly = logs_df[['Subsystem_encoded', 'SensorValue']]
y_anomaly = logs_df['is_emergency']

# Train-test split for anomaly detection
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_anomaly, y_anomaly, test_size=0.3, random_state=42
)

# Random Forest Classifier for anomaly detection
anomaly_clf = RandomForestClassifier(random_state=42)
anomaly_clf.fit(X_train_a, y_train_a)

# Predictions and evaluation
y_pred = anomaly_clf.predict(X_test_a)
print("\nSupervised Anomaly Detection Performance:")
print(classification_report(y_test_a, y_pred))
print(f"Accuracy: {accuracy_score(y_test_a, y_pred):.2f}")

# Feature importance visualization
feature_importances = pd.Series(anomaly_clf.feature_importances_,
                                index=X_anomaly.columns)
plt.figure(figsize=(8, 4))
feature_importances.sort_values().plot(kind='barh')
plt.title('Feature Importances for Anomaly Detection')
plt.show()

# Predictive Maintenance using Rolling Average
logs_df['rolling_avg'] = logs_df['SensorValue'].rolling(window=5).mean()
logs_df['maintenance_flag'] = np.where(logs_df['rolling_avg'] > 125, 'Needs Maintenance', 'OK')

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Anomaly Detection Visualization
pd.Series(y_test_a).value_counts().plot(kind='bar', ax=ax1)
ax1.set_title('Actual Emergency Distribution in Test Set')
ax1.set_ylabel('Count')

# Predictive Maintenance Visualization
logs_df['maintenance_flag'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_title('Predictive Maintenance Flag')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.show()

# Display preview of logs with maintenance flags
print(logs_df[['Subsystem', 'SensorValue', 'AlarmLevel', 'rolling_avg', 'maintenance_flag']].head())