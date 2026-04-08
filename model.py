import pandas as pd
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -------- FILE PATH --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "hostel_students_dataset.csv")


# -------- LOAD DATA --------
def load_data():
    return pd.read_csv(DATA_PATH)


# -------- TRAIN KNN --------
def train_knn(df):
    features = df.drop(columns=["student_id"])
    model = NearestNeighbors(n_neighbors=4)
    model.fit(features)
    return model, features


# -------- TRAIN LOGISTIC REGRESSION --------
def train_lr(df):

    features = df.drop(columns=["student_id"])
    features = features.apply(pd.to_numeric, errors='coerce').fillna(features.mean())

    X = []
    y = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            s1 = features.iloc[i]
            s2 = features.iloc[j]

            diff = abs(s1 - s2)
            X.append(diff.values)

            if diff.mean() < 2:
                y.append(1)
            else:
                y.append(0)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler


# -------- PRIORITY SCORE --------
def calculate_priority_score(s1, s2):

    features = [
        "sleep_time","wake_time","study_hours",
        "silence_preference","cleanliness_level",
        "introversion","emotional_stability","agreeableness"
    ]

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

    distance = total_score / total_weight

    # 🔥 Tinder-style score
    score = 95 - (distance * 5)
    score = max(70, min(score, 95))

    return score


# -------- ML SCORE --------
def calculate_ml_score(model, scaler, s1, s2):

    arr1 = s1.drop("student_id").astype(float).values
    arr2 = s2.drop("student_id").astype(float).values

    diff = abs(arr1 - arr2).reshape(1, -1)
    diff_scaled = scaler.transform(diff)

    prob = model.predict_proba(diff_scaled)[0][1]

    return prob * 100


# -------- AI EXPLANATION --------
def generate_explanation(s1, s2):

    explanation = []

    features_map = {
        "sleep_time": "sleep schedule",
        "wake_time": "wake-up time",
        "study_hours": "study habits",
        "silence_preference": "silence preference",
        "cleanliness_level": "cleanliness",
        "introversion": "personality",
        "emotional_stability": "emotional stability",
        "agreeableness": "nature"
    }

    for key, label in features_map.items():

        diff = abs(float(s1[key]) - float(s2[key]))

        if diff <= 1:
            explanation.append(f"✔ Similar {label}")
        elif diff >= 4:
            explanation.append(f"⚠ Very different {label}")

    return explanation


# -------- FIND TOP 3 MATCHES --------
def find_best_match(df, student_id):

    knn_model, features = train_knn(df)
    lr_model, scaler = train_lr(df)

    index = df[df["student_id"].astype(str) == str(student_id)].index[0]

    distances, indices = knn_model.kneighbors([features.iloc[index]])

    s1 = df.iloc[index]

    matches = []

    for idx in indices[0][1:4]:  # TOP 3

        s2 = df.iloc[idx]

        priority_score = calculate_priority_score(s1, s2)
        ml_score = calculate_ml_score(lr_model, scaler, s1, s2)

        final_score = round((priority_score + ml_score) / 2, 2)

        matches.append({
            "partner": s2.to_dict(),
            "score": final_score,
            "explanation": generate_explanation(s1, s2)
        })

    return {
        "your_data": s1.to_dict(),
        "matches": matches
    }


# -------- ADD STUDENT --------
def add_student_and_find_match(new_student):

    df = load_data()

    if new_student["student_id"] in df["student_id"].astype(str).values:
        return {"error": "Student ID already exists!"}

    df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    return find_best_match(df, new_student["student_id"])


# -------- GET EXISTING --------
def get_match_for_student(student_id):

    df = load_data()

    if student_id not in df["student_id"].astype(str).values:
        return None

    return find_best_match(df, student_id)


# -------- COMPARE TWO STUDENTS --------
def compare_two_students(id1, id2):

    df = load_data()

    if id1 not in df["student_id"].astype(str).values or \
       id2 not in df["student_id"].astype(str).values:
        return None

    s1 = df[df["student_id"].astype(str) == str(id1)].iloc[0]
    s2 = df[df["student_id"].astype(str) == str(id2)].iloc[0]

    lr_model, scaler = train_lr(df)

    priority_score = calculate_priority_score(s1, s2)
    ml_score = calculate_ml_score(lr_model, scaler, s1, s2)

    final_score = round((priority_score + ml_score) / 2, 2)

    return {
        "your_data": s1.to_dict(),
        "partner_data": s2.to_dict(),
        "score": final_score,
        "prediction": "Compatible ✅" if final_score > 75 else "Not Compatible ❌"
    }