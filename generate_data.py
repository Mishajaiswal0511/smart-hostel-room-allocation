import pandas as pd
import numpy as np

data = []

for i in range(1, 101):
    row = [
        i,
        np.random.randint(0,24),   # sleep
        np.random.randint(0,24),   # wake
        np.random.randint(1,10),   # study
        np.random.randint(1,10),
        np.random.randint(1,10),
        np.random.randint(1,10),
        np.random.randint(1,10),
        np.random.randint(1,10),

        # priorities (1–5)
        np.random.randint(1,6),
        np.random.randint(1,6),
        np.random.randint(1,6),
        np.random.randint(1,6),
        np.random.randint(1,6),
        np.random.randint(1,6),
        np.random.randint(1,6),
        np.random.randint(1,6)
    ]
    data.append(row)

columns = [
    "student_id","sleep_time","wake_time","study_hours",
    "silence_preference","cleanliness_level",
    "introversion","emotional_stability","agreeableness",
    "sleep_p","wake_p","study_p","silence_p",
    "clean_p","intro_p","emotion_p","agree_p"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("hostel_students_dataset.csv", index=False)

print("✅ New dataset created!")