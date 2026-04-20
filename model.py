import pandas as pd
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import math

# -------- FILE PATH --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "hostel_students_dataset.csv")

# -------- SCHEMA --------
FEATURE_COLS = [
    "sleep_time", "wake_time", "study_hours",
    "silence_preference", "cleanliness_level",
    "introversion", "emotional_stability", "agreeableness",
]
PRIORITY_COLS = [
    "sleep_p", "wake_p", "study_p",
    "silence_p", "clean_p",
    "intro_p", "emotion_p", "agree_p",
]
ALL_NUMERIC_COLS = FEATURE_COLS + PRIORITY_COLS
STRING_COLS = ["name", "description"]


# -------- GLOBAL CACHE --------
_df = None
_knn_model = None
_knn_features = None
_lr_model = None
_scaler = None


def _is_nan(x) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _safe_number(x, default=0):
    if x is None:
        return default
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            x = x.item()
        except Exception:
            pass
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _clean_record_dict(d: dict) -> dict:
    """
    Ensure a student dict is safe for templates and JSON:
    - no NaN/inf
    - name always string
    - student_id always int-like when possible
    """
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = None
        if k == "name":
            out[k] = "" if v is None else str(v)
        elif k == "student_id":
            try:
                out[k] = int(v)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist
    if "student_id" not in df.columns:
        df["student_id"] = pd.Series(dtype="int64")
    # Student ID should be numeric; keep rows that have a usable id
    df["student_id"] = pd.to_numeric(df["student_id"], errors="coerce")
    df = df[~df["student_id"].isna()].copy()
    df["student_id"] = df["student_id"].astype(int)

    # Numeric columns: create missing, coerce numeric, fill defaults
    for col in ALL_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaNs with sensible defaults
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(0).astype(float)
    for col in PRIORITY_COLS:
        # priorities are 1..5 ideally; default to 3
        df[col] = df[col].fillna(3).astype(float)

    # String columns
    for col in STRING_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
        df.loc[df[col].str.lower().isin(["nan", "none"]), col] = ""

    return df


def get_models(retrain=False):
    global _df, _knn_model, _knn_features, _lr_model, _scaler
    if _df is None or retrain:
        _df = _normalize_df(pd.read_csv(DATA_PATH))
    if _lr_model is None or retrain:
        _knn_model, _knn_features = train_knn(_df)
        _lr_model, _scaler = train_lr(_df)
    return _knn_model, _knn_features, _lr_model, _scaler

# -------- LOAD DATA --------


def load_data():
    global _df
    if _df is not None:
        return _df
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    _df = _normalize_df(pd.read_csv(DATA_PATH))
    return _df


# -------- TRAIN KNN --------
def train_knn(df):
    # Only use numeric columns for training features
    features = df[ALL_NUMERIC_COLS].copy()
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    n_neighbors = 4
    if len(features) <= 1:
        # Not enough data to train meaningful neighbor model
        n_neighbors = 1
    else:
        n_neighbors = min(n_neighbors, len(features))

    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(features)
    return model, features


# -------- TRAIN LOGISTIC REGRESSION --------
def train_lr(df):
    # Only use numeric columns
    features = df[ALL_NUMERIC_COLS].copy()
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

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
        "sleep_time", "wake_time", "study_hours",
        "silence_preference", "cleanliness_level",
        "introversion", "emotional_stability", "agreeableness"
    ]

    priorities = [
        "sleep_p", "wake_p", "study_p",
        "silence_p", "clean_p",
        "intro_p", "emotion_p", "agree_p"
    ]

    total_score = 0
    total_weight = 0

    for f, p in zip(features, priorities):
        diff = abs(_safe_number(s1.get(f) if hasattr(s1, "get") else s1[f]) - _safe_number(s2.get(f) if hasattr(s2, "get") else s2[f]))
        weight = (_safe_number(s1.get(p) if hasattr(s1, "get") else s1[p], default=3) + _safe_number(s2.get(p) if hasattr(s2, "get") else s2[p], default=3)) / 2

        total_score += diff * weight
        total_weight += weight

    if total_weight <= 0:
        return 70.0
    distance = total_score / total_weight

    # 🔥 Tinder-style score
    score = 95 - (distance * 5)
    score = max(70, min(score, 95))

    return score


# -------- ML SCORE --------
def calculate_ml_score(model, scaler, s1, s2):

    a1 = s1[ALL_NUMERIC_COLS]
    a2 = s2[ALL_NUMERIC_COLS]
    a1 = pd.to_numeric(a1, errors="coerce").fillna(0)
    a2 = pd.to_numeric(a2, errors="coerce").fillna(0)
    arr1 = a1.astype(float).values
    arr2 = a2.astype(float).values

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

        diff = abs(_safe_number(s1.get(key) if hasattr(s1, "get") else s1[key]) - _safe_number(s2.get(key) if hasattr(s2, "get") else s2[key]))

        if diff <= 1:
            explanation.append(f"✔ Similar {label}")
        elif diff >= 4:
            explanation.append(f"⚠ Very different {label}")

    return explanation


# -------- FIND TOP 3 MATCHES --------
def find_best_match(df, student_id):

    knn_model, features, lr_model, scaler = get_models()

    df = _normalize_df(df)
    if len(df) == 0:
        return None

    row_idx = df[df["student_id"].astype(str) == str(student_id)].index
    if len(row_idx) == 0:
        return None
    index = row_idx[0]

    # Use a 1-row DataFrame (keeps feature names) to avoid sklearn warnings
    distances, indices = knn_model.kneighbors(features.iloc[[index]])

    s1 = df.iloc[index]

    matches = []

    # when dataset is small, indices may not have 4 neighbors
    for idx in list(indices[0])[1:4]:  # TOP 3

        s2 = df.iloc[idx]

        priority_score = calculate_priority_score(s1, s2)
        ml_score = calculate_ml_score(lr_model, scaler, s1, s2)

        final = (float(priority_score) + float(ml_score)) / 2
        if math.isnan(final) or math.isinf(final):
            final = 70.0
        final_score = round(final, 2)

        matches.append({
            "partner": _clean_record_dict(s2.to_dict()),
            "score": final_score,
            "explanation": generate_explanation(s1, s2)
        })

    return {
        "your_data": _clean_record_dict(s1.to_dict()),
        "matches": matches
    }


# -------- ADD STUDENT --------
def get_all_students():
    df = load_data()
    records = df.to_dict(orient="records")
    return [_clean_record_dict(r) for r in records]


def delete_student(student_id):
    global _df
    df = load_data()
    df = df[df["student_id"].astype(str) != str(student_id)]
    df.to_csv(DATA_PATH, index=False)
    _df = df
    get_models(retrain=True)
    return True


def update_student(student_id, updated_data):
    global _df
    df = load_data()
    if not (df["student_id"].astype(str) == str(student_id)).any():
        return {"error": "❌ Student ID not found!"}

    for key, val in updated_data.items():
        if key in df.columns and key != "student_id":
            if key in STRING_COLS:
                df.loc[df["student_id"].astype(str) == str(student_id), key] = str(val) if val else ""
            else:
                try:
                    df.loc[df["student_id"].astype(str) == str(student_id), key] = int(float(val))
                except (ValueError, TypeError):
                    pass

    df = _normalize_df(df)
    df.to_csv(DATA_PATH, index=False)
    _df = df
    get_models(retrain=True)
    return {"success": True}


def add_student_and_find_match(new_student):
    df = load_data()

    if "student_id" not in new_student:
        return {"error": "❌ Student ID is required"}

    try:
        new_student["student_id"] = int(new_student["student_id"])
    except (ValueError, TypeError):
        return {"error": "❌ Invalid Student ID. Must be an integer."}

    if str(new_student["student_id"]) in df["student_id"].astype(str).values:
        return {"error": "❌ Student ID already exists in the database"}

    # ensure missing fields don't become NaN in the CSV
    base = {c: 0 for c in ALL_NUMERIC_COLS}
    base["name"] = ""
    payload = {**base, **(new_student or {})}
    df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
    df = _normalize_df(df)
    df.to_csv(DATA_PATH, index=False)

    # Force cache refresh globally
    global _df
    _df = df
    get_models(retrain=True)

    return find_best_match(df, new_student["student_id"])


# -------- GET EXISTING --------
def get_match_for_student(student_id):

    df = load_data()

    if str(student_id) not in df["student_id"].astype(str).values:
        return None

    return find_best_match(df, student_id)


# -------- COMPARE TWO STUDENTS --------
def compare_two_students(id1, id2):

    df = load_data()

    if str(id1) not in df["student_id"].astype(str).values or \
       str(id2) not in df["student_id"].astype(str).values:
        return None

    s1 = df[df["student_id"].astype(str) == str(id1)].iloc[0]
    s2 = df[df["student_id"].astype(str) == str(id2)].iloc[0]

    _, _, lr_model, scaler = get_models()

    priority_score = calculate_priority_score(s1, s2)
    ml_score = calculate_ml_score(lr_model, scaler, s1, s2)

    final_score = round((priority_score + ml_score) / 2, 2)

    return {
        "your_data": s1.to_dict(),
        "partner_data": s2.to_dict(),
        "score": final_score,
        "explanation": generate_explanation(s1, s2),
        "prediction": "Compatible ✅" if final_score > 75 else "Not Compatible ❌"
    }
