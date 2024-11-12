import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load dataset
data = pd.read_excel('DatasetDummy.xlsx')

# Rename columns for easier access
data.columns = ["Index", "Age", "Height_cm", "Weight_hg", "Bmi", "BmiClass"]
data = data.drop(0).reset_index(drop=True)

# Convert height and weight to appropriate units
data['Height_m'] = data['Height_cm'].astype(float) / 100
data['Weight_kg'] = data['Weight_hg'].astype(float) / 10

# Recalculate BMI
data['Bmi'] = data['Weight_kg'] / (data['Height_m'] ** 2)

# Prepare features and target
X = data[['Age', 'Height_cm', 'Weight_hg']].astype(float)
y = data['BmiClass']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Function to calculate BMI and classify
def calculate_bmi_and_classify(age, height_cm, weight_hg):
    height_m = height_cm / 100
    weight_kg = weight_hg / 10
    bmi = weight_kg / (height_m ** 2)

    # Prepare input for prediction
    input_data = pd.DataFrame([[age, height_cm, weight_hg]], columns=['Age', 'Height_cm', 'Weight_hg'])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict class
    bmi_class = knn.predict(input_data_scaled)[0]
    
    return bmi, bmi_class, accuracy, classification_rep

# Example usage
example_bmi, example_class, model_accuracy, model_report = calculate_bmi_and_classify(25, 180, 750)
print("BMI:", example_bmi)
print("BMI Class:", example_class)
print("Model Accuracy:", model_accuracy)
print("Classification Report:\n", model_report)

# Input dari pengguna
age = int(input("Masukkan umur (tahun): "))
height_cm = float(input("Masukkan tinggi badan (cm): "))
weight_hg = float(input("Masukkan berat badan (hg): "))

# Menggunakan model untuk menghitung dan mengklasifikasi BMI
bmi, bmi_class, model_accuracy, model_report = calculate_bmi_and_classify(age, height_cm, weight_hg)

# Menampilkan hasil
print("\nHasil Penghitungan:")
print("BMI:", bmi)
print("Klasifikasi BMI:", bmi_class)
print("\nAkurasi Model:", model_accuracy)
print("\nLaporan Klasifikasi:\n", model_report)

