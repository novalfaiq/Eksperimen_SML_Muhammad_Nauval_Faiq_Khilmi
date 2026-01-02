import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import argparse

def run_preprocessing(input_path, output_dir):
    """
    Fungsi utama untuk mengotomatisasi pembersihan dan persiapan data.
    """
    print(f"--- Memulai proses otomatisasi untuk: {input_path} ---")
    
    # 1. Memuat Dataset
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return
    
    df = pd.read_csv(input_path)
    
    # 2. Pembersihan Data (Cleaning)
    # Menghapus student_id karena tidak memiliki nilai prediktif
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])
        print("- Kolom 'student_id' berhasil dihapus.")

    # 3. Encoding Data Kategorikal
    # Mengubah teks menjadi angka agar bisa diproses model ML
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    print(f"- Encoding selesai untuk kolom: {list(categorical_cols)}")

    # 4. Memisahkan Fitur dan Target
    X = df.drop(columns=['exam_score'])
    y = df['exam_score']

    # 5. Standarisasi/Scaling
    # Menyamakan skala fitur numerik agar model lebih stabil
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("- Standarisasi fitur (Scaling) selesai.")

    # 6. Split Dataset
    # Membagi 80% untuk latihan dan 20% untuk pengujian
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 7. Menyimpan Hasil (Exporting)
    # Membuat folder tujuan jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"--- SELESAI: Data siap latih disimpan di folder '{output_dir}' ---")

if __name__ == "__main__":
    # Menggunakan argparse agar bisa dijalankan lewat terminal dengan parameter fleksibel
    parser = argparse.ArgumentParser(description="Otomatisasi Preprocessing Dataset Exam Score")
    parser.add_argument('--input', type=str, default='exam_score_prediction_raw.csv', help='Path ke file raw data')
    parser.add_argument('--output', type=str, default='preprocessing/exam_score_prediction_preprocessing', help='Folder tujuan hasil preprocessing')
    
    args = parser.parse_args()
    
    run_preprocessing(args.input, args.output)