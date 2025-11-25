from fastapi import FastAPI
from db import get_connection
import numpy as np
import joblib
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "risk_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "risk_scaler.pkl"))


# =============== DB 조회 ===============
def get_user_base(user_id):
    conn = get_connection()
    cur = conn.cursor()

    sql = """
        SELECT USER_ID, AGE, GENDER, HEIGHT, CURRENT_WEIGHT
        FROM USERS
        WHERE USER_ID = %s
    """
    cur.execute(sql, (user_id,))
    result = cur.fetchone()

    cur.close()
    conn.close()
    return result


def get_today_nutrition(user_id):
    conn = get_connection()
    cur = conn.cursor()

    sql = """
        SELECT
            IFNULL(SUM(CALORIES_KCAL), 0) AS total_calories,
            IFNULL(SUM(CARBS_G), 0) AS total_carbs,
            IFNULL(SUM(PROTEIN_G), 0) AS total_protein,
            IFNULL(SUM(FATS_G), 0) AS total_fat,
            IFNULL(SUM(SUGAR_G), 0) AS total_sugar,
            IFNULL(SUM(SODIUM_MG), 0) AS total_sodium
        FROM MEAL_ENTRY
        WHERE USER_ID = %s
        AND DATE(EAT_TIME) = CURDATE()
    """

    cur.execute(sql, (user_id,))
    result = cur.fetchone()

    cur.close()
    conn.close()
    return result


# =============== 입력 벡터 생성 ===============
def make_input_vector(base, nutri):

    # 만약 음식 칼로리가 0이면 즉시 위험도 0 처리
    if nutri["total_calories"] == 0:
        return None  # → 모델로 안 보내고 바로 0% return

    sex = 1 if base["GENDER"] == "M" else 0
    age = base["AGE"]

    total_cal = nutri["total_calories"]
    carbs = nutri["total_carbs"]
    protein = nutri["total_protein"]
    fat = nutri["total_fat"]

    if total_cal > 0:
        carb_ratio = carbs * 4 / total_cal * 100
        protein_ratio = protein * 4 / total_cal * 100
        fat_ratio = fat * 9 / total_cal * 100
    else:
        carb_ratio = protein_ratio = fat_ratio = 0

    return np.array([[
        sex, age,
        total_cal,
        carb_ratio, protein_ratio, fat_ratio,
        nutri["total_sugar"],
        nutri["total_sodium"]
    ]])


# =============== 실제 API ===============
@app.get("/predict-risk/{user_id}")
def predict_risk(user_id: int):
    base = get_user_base(user_id)
    nutri = get_today_nutrition(user_id)

    if base is None:
        return {"error": "유저 없음"}

    # 1) 음식 기록이 아예 없으면 위험도 0%
    if nutri["total_calories"] == 0:
        return {
            "user_id": user_id,
            "risk_score": 0.0
        }

    # 2) 모델 입력 생성
    x = make_input_vector(base, nutri)

    # x가 None이면 음식 सेवन 없음
    if x is None:
        return {
            "user_id": user_id,
            "risk_score": 0.0
        }

    # 3) 모델 예측
    x_scaled = scaler.transform(x)
    risk = model.predict(x_scaled)[0]

    # 4) 0~100 보정
    risk = max(0, min(100, risk))

    return {
        "user_id": user_id,
        "risk_score": float(risk)
    }
