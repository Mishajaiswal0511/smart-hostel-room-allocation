import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegression

# -------- FILE PATH --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "hostel_students_dataset.csv")


# -------- ADD NEW STUDENT --------
def add_student_and_find_match(new_student):
    df = pd.read_csv(DATA_PATH)

    if 'cluster' in df.columns:
        df = df.drop(columns=['cluster'])

    if new_student["student_id"] in df["student_id"].astype(str).values:
        return {"error": "Student ID already exists!"}

    df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    return find_best_match(df, new_student["student_id"])


# -------- GET EXISTING MATCH --------
def get_match_for_student(student_id):
    df = pd.read_csv(DATA_PATH)

    if 'cluster' in df.columns:
        df = df.drop(columns=['cluster'])

    if student_id not in df["student_id"].astype(str).values:
        return None

    return find_best_match(df, student_id)


# -------- SINGLE STUDENT MATCH --------
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

    # 🎯 Score (smooth + realistic)
    score = 100 * np.exp(-0.5 * min_distance)

    if score < 40:
        score += 40

    score = round(score, 2)

    return {
        "your_data": df.iloc[index].to_dict(),
        "partner_data": df.iloc[partner_index].to_dict(),
        "score": score
    }


# -------- TRAIN ML MODEL --------
def train_model(df):

    pairs = []
    labels = []

    data = df.drop(columns=["student_id"])
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(data.mean())

    for i in range(len(data)):
        for j in range(i + 1, len(data)):

            s1 = data.iloc[i]
            s2 = data.iloc[j]

            diff = abs(s1 - s2)
            pairs.append(diff.values)

            # Label rule
            if diff.mean() < 2:
                labels.append(1)
            else:
                labels.append(0)

    X = np.array(pairs)
    y = np.array(labels)

    model = LogisticRegression()
    model.fit(X, y)

    return model


# -------- PREDICT COMPATIBILITY --------
def predict_compatibility(model, student1, student2):

    s1 = np.array(list(student1.values())[1:], dtype=float)
    s2 = np.array(list(student2.values())[1:], dtype=float)

    diff = abs(s1 - s2).reshape(1, -1)

    prob = model.predict_proba(diff)[0][1]

    return round(prob * 100, 2)


# -------- COMPARE TWO STUDENTS (WITH WEIGHTS) --------
def compare_two_students(id1, id2):

    df = pd.read_csv(DATA_PATH)

    if id1 not in df["student_id"].astype(str).values or \
       id2 not in df["student_id"].astype(str).values:
        return None

    s1 = df[df["student_id"].astype(str) == str(id1)].iloc[0]
    s2 = df[df["student_id"].astype(str) == str(id2)].iloc[0]

    # 🎯 FEATURES
    features = [
        "sleep_time","wake_time","study_hours",
        "silence_preference","cleanliness_level",
        "introversion","emotional_stability","agreeableness"
    ]

    # 🎯 PRIORITIES
    priorities = [
        "sleep_p","wake_p","study_p",
        "silence_p","clean_p",
        "intro_p","emotion_p","agree_p"
    ]

    total_score = 0
    total_weight = 0

    for f, p in zip(features, priorities):

        diff = abs(float(s1[f]) - float(s2[f]))

        weight = (float(s1[p]) + float(s2[p])) / 2

        total_score += diff * weight
        total_weight += weight

    # Normalize
    final_distance = total_score / total_weight

    # Convert to % score
    score = 100 * np.exp(-0.5 * final_distance)

    if score < 40:
        score += 40

    score = round(score, 2)

    return {
        "your_data": s1.to_dict(),
        "partner_data": s2.to_dict(),
        "score": score,
         "features": features,
    "differences": [
        abs(float(s1[f]) - float(s2[f])) for f in features
    ]
        
    }

    # -------- FINAL COMBINED SCORE --------
    final_score = round((similarity_score + ml_score) / 2, 2)

    return {
        "your_data": student1.to_dict(),
        "partner_data": student2.to_dict(),
        "score": final_score,
        "prediction": "Compatible ✅" if final_score > 60 else "Not Compatible ❌"
    }