from flask import Flask, render_template, request
from model import add_student_and_find_match, get_match_for_student

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
            "agreeableness": int(request.form['agreeableness'])
        }

        # Backend validation
        for key, value in new_student.items():
            if key != "student_id":
                if int(value) < 0:
                    return "❌ Only positive integers allowed!"

    except:
        return "❌ Invalid input!"

    result = add_student_and_find_match(new_student)

# Handle duplicate error
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
        return render_template("result.html", error="❌ Student ID not found in database!")


# -------- RUN --------
if __name__ == "__main__":
    app.run(debug=True)