# -*- coding: utf-8 -*-
"""Human Resources.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jeF32uTDIKldl2I8XjQnVZ_twTnKnuaf

# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

- Nama: Muhammad Farhan Juna
- Email: muhammadfarhan.mf711@gmail.com
- Id Dicoding: mjuna

## Persiapan

### Menyiapkan library yang dibutuhkan
"""

# Mengimpor library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

"""### Menyiapkan data yang akan digunakan"""

# Memuat data
data = pd.read_csv('/content/employee_data.csv')

"""## Data Understanding"""

# Melihat informasi awal tentang data
data.info()

# Sample Head Data
data.head()

# Melihat statistik deskriptif dari data
data.describe()

"""## Data Preparation / Preprocessing"""

# Mengecek apakah ada missing values
print(data.isnull().sum())

# Mengisi missing values dengan nilai modus (untuk data kategorikal) atau median (untuk data numerik)
data['Attrition'].fillna(data['Attrition'].mode()[0])

# Mengecek apakah ada missing values
print(data.isnull().sum())

import pandas as pd
from sqlalchemy import create_engine

# Send Dataset to Supabase
DATABASE_URL = "postgresql://postgres.rijjukdsjzammsxpaznc:C0hNrJiwrrMc7qys@aws-0-us-east-1.pooler.supabase.com:6543/postgres"

# Create a database engine
engine = create_engine(DATABASE_URL)

# Assuming 'data' is your pandas DataFrame from the previous code
# Example usage with your provided DataFrame
try:
  data.to_sql('your_table_name', engine, if_exists='replace', index=False)  # Replace 'your_table_name' with the desired table name
  print("Data successfully sent to Supabase")
except Exception as e:
  print(f"An error occurred: {e}")

"""# Data Visualizaztion

2. Analisis Univariate

Distribusi Data Numerik
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Distribusi Usia Karyawan')
plt.xlabel('Usia')
plt.ylabel('Frekuensi')
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=data)
plt.title('Box Plot Pendapatan Bulanan berdasarkan Attrition')
plt.xlabel('Attrition')
plt.ylabel('Pendapatan Bulanan')
plt.show()

"""Distribusi Data Kategorikal"""

plt.figure(figsize=(8, 5))
sns.countplot(x='Attrition', data=data)
plt.title('Distribusi Attrition')
plt.xlabel('Attrition')
plt.ylabel('Jumlah')
plt.show()

"""Insight dari Analisis Univariate:

Distribusi Usia Karyawan:
- Dari histogram Age, terlihat bahwa sebagian besar karyawan berusia antara 30-40 tahun. Distribusi data cenderung normal (bell-shaped) tetapi sedikit miring ke kanan (right-skewed) menandakan ada beberapa karyawan yang berusia lebih tua.
- Distribusi Attrition: Dari count plot Attrition, terlihat bahwa sebagian besar karyawan tidak mengalami attrition (No). Namun, ada sejumlah karyawan yang mengalami attrition (Yes), yang menjadi fokus perhatian kita.
- Box Plot Pendapatan Bulanan berdasarkan Attrition: Karyawan yang mengalami attrition cenderung memiliki pendapatan bulanan yang lebih rendah dibandingkan dengan karyawan yang tidak mengalami attrition. Terdapat outlier pada karyawan yang tidak mengalami attrition (No), menandakan ada beberapa karyawan dengan pendapatan bulanan yang sangat tinggi.
-Distribusi Data Kategorikal lainnya: Dari count plot variabel kategorikal lainnya seperti BusinessTravel, Department, Gender, dll., kita bisa melihat distribusi masing-masing kategori dan mengidentifikasi kategori yang dominan atau minoritas.

Analisis Bivariate

Korelasi antara Variabel Numerik
"""

plt.figure(figsize=(12, 8))
sns.heatmap(data[['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome']].corr(), annot=True, cmap='coolwarm')
plt.title('Korelasi antara Variabel Numerik')
plt.show()

"""Hubungan antara Variabel Kategorikal"""

pd.crosstab(data['Attrition'], data['Department']).plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Hubungan antara Attrition dan Department')
plt.xlabel('Attrition')
plt.ylabel('Jumlah')
plt.show()

"""Insight dari Analisis Bivariate:

- Korelasi antara Variabel Numerik: Dari heatmap, terlihat bahwa ada korelasi positif yang moderat antara Age dan MonthlyIncome. Artinya, semakin tua usia karyawan, cenderung semakin tinggi pendapatan bulanannya.
- Hubungan antara Attrition dan Pendapatan Bulanan: Dari box plot, kita sudah melihat bahwa karyawan yang mengalami attrition cenderung memiliki pendapatan bulanan yang lebih rendah. Hal ini menunjukkan bahwa pendapatan bulanan bisa menjadi salah satu faktor yang mempengaruhi attrition.
- Hubungan antara Attrition dan Department: Dari stacked bar chart, terlihat bahwa proporsi karyawan yang mengalami attrition berbeda-beda di setiap department. Misalnya, department Sales dan Research & Development memiliki proporsi attrition yang relatif lebih tinggi dibandingkan dengan department Human Resources.
- Hubungan antara Variabel Kategorikal dan Numerik lainnya: Dari box plot atau violin plot lainnya, kita bisa melihat bagaimana variabel numerik terdistribusi di setiap kategori variabel kategorikal. Ini dapat memberikan insight tentang perbedaan karakteristik antar kategori.
Insight dari Identifikasi Outlier:

- Outlier yang terdeteksi pada variabel numerik seperti MonthlyIncome perlu diinvestigasi lebih lanjut. Outlier ini bisa jadi merupakan data yang valid (misalnya, karyawan dengan posisi tinggi) atau bisa juga merupakan kesalahan input data.

## Modeling & Evaluation

## Random Forest Model
"""

# Menentukan fitur (X) dan target (y)
X = data.drop(columns=['Attrition', 'EmployeeId'])  # Menghapus kolom yang tidak relevan
y = data['Attrition']

# Membagi data menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Menentukan preprocessor untuk menangani fitur kategorikal dan numerik
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome']),
        ('cat', OneHotEncoder(), ['BusinessTravel', 'Department', 'Gender', 'EducationField', 'JobRole', 'MaritalStatus', 'OverTime'])
    ]
)

# Membuat pipeline dengan RandomForestClassifier
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Melatih model
pipeline_rf.fit(X_train, y_train)

# Prediksi pada data uji
y_pred_rf = pipeline_rf.predict(X_test)

# Evaluasi model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

import joblib
# Save Modal
joblib.dump(pipeline_rf, 'random_forest_model.pkl')

"""Prediction"""

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Memuat model yang sudah disimpan
model = joblib.load('random_forest_model.pkl')

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

"""## Logistic Regression Model"""

# Membuat pipeline dengan LogisticRegression
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Melatih model
pipeline_lr.fit(X_train, y_train)

# Prediksi pada data uji
y_pred_lr = pipeline_lr.predict(X_test)

# Evaluasi model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))

# Save the LogisticRegression model
joblib.dump(pipeline_lr, 'logistic_regression_model.pkl')

"""Prediction"""

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Memuat model yang sudah disimpan
model = joblib.load('logistic_regression_model.pkl')

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

