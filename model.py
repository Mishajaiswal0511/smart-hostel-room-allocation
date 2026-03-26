import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "hostel_students_dataset.csv")


def add_student_and_find_match(new_student):
    df = pd.read_csv(DATA_PATH)

    if 'cluster' in df.columns:
        df = df.drop(columns=['cluster'])

    if new_student["student_id"] in df["student_id"].astype(str).values:
        return {"error": "Student ID already exists!"}

    df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    return find_best_match(df, new_student["student_id"])


def get_match_for_student(student_id):
    df = pd.read_csv(DATA_PATH)

    if 'cluster' in df.columns:
        df = df.drop(columns=['cluster'])

    if student_id not in df["student_id"].astype(str).values:
        return None

    return find_best_match(df, student_id)


def find_best_match(df, student_id):

    features = df.drop(columns=["student_id"])
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.fillna(features.mean())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    distances = euclidean_distances(scaled, scaled)

    index = df[df["student_id"].astype(str) == str(student_id)].index[0]
    distances[index][index] = float("inf")

    partner_index = distances[index].argmin()
    min_distance = distances[index][partner_index]

    # 🎯 FINAL SCORE (balanced + user friendly)
    score = 100 * np.exp(-0.5 * min_distance)

    # UX adjustment (avoid too low values)
    if score < 40:
        score += 40

    score = round(score, 2)

    return {
        "your_data": df.iloc[index].to_dict(),
        "partner_data": df.iloc[partner_index].to_dict(),
        "score": score
    }