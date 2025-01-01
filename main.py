import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer  # For handling missing values

# Load all datasets
data1 = pd.read_csv('bank-full.csv', sep=';')
data2 = pd.read_csv('bank.csv', sep=';')
data3 = pd.read_csv('bank-additional-full.csv', sep=';')
data4 = pd.read_csv('bank-additional.csv', sep=';')

# Combine datasets vertically (assuming similar structures)
data = pd.concat([data1, data2, data3, data4], ignore_index=True)

# Display dataset info
print("Combined Dataset Info:")
print(data.info())

# Identify categorical and numerical columns
categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
numerical_cols = [col for col in data.columns if col not in categorical_cols and col != 'y']

# Handle missing values
# For numerical columns: Fill missing values with the median
imputer_num = SimpleImputer(strategy='median')
data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])

# For categorical columns: Fill missing values with the most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col].astype(str))  # Convert to string and then encode

# Feature scaling for numerical variables
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Define features (X) and target (y)
X = data.drop('y', axis=1)  # Features
y = data['y']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Random Forest Evaluation
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))

# Neural Network Classifier
print("\nTraining Neural Network...")
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
nn_model.fit(X_train, y_train)
nn_predictions = nn_model.predict(X_test)

# Neural Network Evaluation
print("\nNeural Network Results:")
print("Accuracy:", accuracy_score(y_test, nn_predictions))
print("Classification Report:\n", classification_report(y_test, nn_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, nn_predictions))
