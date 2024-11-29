import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Memuat model yang sudah disimpan
model = joblib.load('random_forest_model.pkl') 
# model = joblib.load('logistic_regression_model.pkl')

# Menyediakan data baru yang ingin diprediksi (misalnya data karyawan baru)
data_new = pd.DataFrame({
    'Age': [45],
    'DailyRate': [2000],
    'DistanceFromHome': [10],
    'HourlyRate': [50],
    'JobInvolvement': [3],
    'JobSatisfaction': [4],
    'MonthlyIncome': [5000],
    'BusinessTravel': ['Travel_Rarely'],
    'Department': ['Sales'],
    'Gender': ['Male'],
    'EducationField': ['Life Sciences'],
    'JobRole': ['Sales Executive'],
    'MaritalStatus': ['Single'],
    'OverTime': ['Yes']
})

# Menentukan preprocessor untuk menangani fitur kategorikal dan numerik
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome']),
        ('cat', OneHotEncoder(), ['BusinessTravel', 'Department', 'Gender', 'EducationField', 'JobRole', 'MaritalStatus', 'OverTime'])
    ]
)

# Menggunakan model untuk prediksi
prediksi = model.predict(data_new)

# Menampilkan hasil prediksi
if prediksi[0] == 1:
    print("Karyawan diprediksi akan meninggalkan perusahaan (Attrition).")
else:
    print("Karyawan diprediksi tidak akan meninggalkan perusahaan.")
