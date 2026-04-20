from flask import Flask, render_template, request, redirect, url_for, session
from flask import Response
import json
import math
from model import (
    add_student_and_find_match,
    get_match_for_student,
    compare_two_students,
    delete_student as m_delete_student,
    get_all_students,
    update_student as m_update_student
)

app = Flask(__name__)
app.secret_key = "premium_key"

def _sanitize_for_json(obj):
    """
    Convert values that are not valid JSON (NaN/inf, numpy scalars, pandas NA)
    into JSON-safe equivalents (None / primitives).
    """
    # numpy scalars sometimes show up in pandas to_dict()
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _sanitize_for_json(obj.item())
        except Exception:
            pass

    # floats: ban NaN and infinity
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # dict / list recursion
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    # other primitives are fine (including None/bool/int/str)
    return obj


def json_response(payload, status=200):
    safe = _sanitize_for_json(payload)
    return Response(
        json.dumps(safe, ensure_ascii=False, allow_nan=False),
        status=status,
        mimetype="application/json",
    )


@app.route('/')
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))

    role = session.get("role")

    if role == "admin":
        students = get_all_students()
        return render_template("dashboard.html", role="admin", students=students)
    else:
        student_id = session.get("user_id")
        result = get_match_for_student(student_id)
        if not result:
            session.clear()
            return redirect(url_for('login'))
        return render_template("dashboard.html", role="student", data=result)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        login_type = request.form.get("login_type")
        if login_type == "admin":
            admin_id = request.form.get("admin_id")
            password = request.form.get("password")
            if admin_id == "admin" and password == "admin123":
                session["user_id"] = "admin"
                session["role"] = "admin"
                return redirect(url_for("home"))
            return render_template("login.html", error="❌ Invalid Admin Credentials!")
        else:
            student_id = request.form.get("student_id")
            name = request.form.get("name")
            result = get_match_for_student(student_id)
            if result:
                # Optionally update their name in the database
                if name:
                    m_update_student(student_id, {"name": name})
                session["user_id"] = student_id
                session["role"] = "student"
                return redirect(url_for("home"))
            return render_template("login.html", error="❌ Student ID not found!")
    return render_template("login.html")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/favicon.ico")
def favicon():
    # Avoid noisy 404s in dev; add a real icon later if desired
    return "", 204


@app.route('/delete/<student_id>', methods=['POST'])
def delete_student_route(student_id):
    if session.get("role") != "admin":
        return "Unauthorized", 403
    m_delete_student(student_id)
    return redirect(url_for("home"))

# ================= REST APIs (AJAX) =================


@app.route('/update', methods=['POST'])
def update():
    if session.get("role") != "student":
        return json_response({"error": "❌ Unauthorized!"}, status=403)
    try:
        data = request.get_json(silent=True)
        if not data:
            return json_response({"error": "❌ Invalid request data format!"}, status=400)

        student_id = session.get("user_id")
        res = m_update_student(student_id, data)
        status = 400 if isinstance(res, dict) and res.get("error") else 200
        return json_response(res, status=status)
    except Exception as e:
        return json_response({"error": "❌ Internal server error!"}, status=500)


@app.route('/submit', methods=['POST'])
def submit():
    if session.get("role") != "admin":
        return json_response({"error": "❌ Unauthorized! Only Admins can add students."}, status=403)
    try:
        new_student = request.get_json(silent=True)
        if not new_student:
            return json_response({"error": "❌ Invalid request payload"}, status=400)
        res = add_student_and_find_match(new_student)
        if "error" in res:
            return json_response(res, status=400)
        return json_response({"success": True, "message": "✅ Student added successfully!", "data": res})
    except Exception as e:
        return json_response({"error": "❌ Internal server error processing request"}, status=500)


@app.route('/check', methods=['POST'])
def check_existing():
    try:
        data = request.get_json(silent=True)
        if not data:
            return json_response({"error": "❌ Invalid JSON data"}, status=400)

        student_id = data.get('student_id')
        if not student_id:
            return json_response({"error": "❌ Please provide a Student ID"}, status=400)

        result = get_match_for_student(student_id)
        if result:
            return json_response({"success": True, "data": result})
        return json_response({"error": "❌ Student not found in database"}, status=404)
    except Exception as e:
        return json_response({"error": "❌ Server generated an invalid response."}, status=500)


@app.route('/check_two', methods=['POST'])
def check_two():
    try:
        data = request.get_json(silent=True)
        if not data:
            return json_response({"error": "❌ Invalid JSON payload"}, status=400)

        id1 = data.get('student1')
        id2 = data.get('student2')

        if not id1 or not id2:
            return json_response({"error": "❌ Missing one or both ID parameters!"}, status=400)

        if str(id1) == str(id2):
            return json_response({"error": "❌ Please enter two different Student IDs!"}, status=400)

        result = compare_two_students(id1, id2)
        if result:
            return json_response({"success": True, "data": result})
        return json_response({"error": "❌ One or both student IDs not found"}, status=404)
    except Exception as e:
        return json_response({"error": "❌ Internal error calculating match"}, status=500)


if __name__ == "__main__":
    app.run(debug=True)
