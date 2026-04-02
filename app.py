from flask import Flask, render_template, request
from model import (
    add_student_and_find_match,
    get_match_for_student,
    compare_two_students
)

app = Flask(__name__)


# -------- HOME --------
@app.route('/')
def index():
    return render_template("index.html")


# -------- ADD STUDENT --------
@app.route('/submit', methods=['POST'])
def submit():
    try:
        new_student = {
    "student_id": request.form['student_id'],

    "sleep_time": int(request.form['sleep_time']),
    "wake_time": int(request.form['wake_time']),
    "study_hours": int(request.form['study_hours']),
    "silence_preference": int(request.form['silence_preference']),
    "cleanliness_level": int(request.form['cleanliness_level']),
    "introversion": int(request.form['introversion']),
    "emotional_stability": int(request.form['emotional_stability']),
    "agreeableness": int(request.form['agreeableness']),

    # ⭐ PRIORITIES
    "sleep_p": int(request.form['sleep_p']),
    "wake_p": int(request.form['wake_p']),
    "study_p": int(request.form['study_p']),
    "silence_p": int(request.form['silence_p']),
    "clean_p": int(request.form['clean_p']),
    "intro_p": int(request.form['intro_p']),
    "emotion_p": int(request.form['emotion_p']),
    "agree_p": int(request.form['agree_p'])
}

        for key, value in new_student.items():
            if key != "student_id" and int(value) < 0:
                return render_template("result.html", error="❌ Only positive numbers allowed!")

    except:
        return render_template("result.html", error="❌ Invalid input!")

    result = add_student_and_find_match(new_student)

    if "error" in result:
        return render_template("result.html", error=result["error"])

    return render_template("result.html", data=result)


# -------- CHECK EXISTING --------
@app.route('/check', methods=['POST'])
def check_existing():

    student_id = request.form['student_id']
    result = get_match_for_student(student_id)

    if result:
        return render_template("result.html", data=result)
    else:
        return render_template("result.html", error="❌ Student ID not found!")


# -------- CHECK TWO --------
@app.route('/check_two', methods=['POST'])
def check_two():

    try:
        id1 = request.form.get("student1")
        id2 = request.form.get("student2")

        # ❌ Same ID check
        if id1 == id2:
            return render_template("result.html", error="❌ Please enter two different Student IDs!")

    except:
        return render_template("result.html", error="❌ Invalid input!")

    # ✅ NO WEIGHTS HERE NOW
    result = compare_two_students(id1, id2)

    if result is None:
        return render_template("result.html", error="❌ One or both Student IDs not found!")

    return render_template("result.html", data=result)

# ✅ ALWAYS AT BOTTOM
if __name__ == "__main__":
    app.run(debug=True)