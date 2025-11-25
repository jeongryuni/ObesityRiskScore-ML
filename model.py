import pyreadstat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 데이터 로드 (KNHANES 24시간 회상조사)
data = "HN23_ALL.sav"
df_all, meta = pyreadstat.read_sav(data)

selected_columns = [
    "sex", "age",
    "N_EN", "N_CHO", "N_PROT", "N_FAT",
    "N_SUGAR", "N_NA"
]

df = df_all[selected_columns].dropna().copy()

# 칼로리가 0이거나 비상식적인 경우 제거
df = df[df["N_EN"] > 0]

# 2. 영양소 비율 계산
df["carb_ratio"] = df["N_CHO"] * 4 / df["N_EN"] * 100
df["protein_ratio"] = df["N_PROT"] * 4 / df["N_EN"] * 100
df["fat_ratio"] = df["N_FAT"] * 9 / df["N_EN"] * 100

df = df[(df["carb_ratio"] > 0) & (df["carb_ratio"] < 100)]
df = df[(df["protein_ratio"] > 0) & (df["protein_ratio"] < 100)]
df = df[(df["fat_ratio"] > 0) & (df["fat_ratio"] < 100)]

# 위험도 레이블 생성 (0~100)
# 최종 위험도 공식: 음식 구성 기반
df["risk_score"] = (
      (df["N_EN"] / 2500) * 30
    + (df["N_FAT"] / 70) * 25
    + (df["N_SUGAR"] / 50) * 20
    + (df["N_NA"] / 2000) * 25
)

df["risk_score"] = df["risk_score"].clip(0, 100)

# 4. 학습 준비
X = df[[
    "sex", "age",
    "N_EN",
    "carb_ratio", "protein_ratio", "fat_ratio",
    "N_SUGAR",
    "N_NA"
]]

y = df["risk_score"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 스케일링 + 모델 학습
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42
)

model.fit(x_train_scaled, y_train)

# 평가
pred = model.predict(x_test_scaled)
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# 모델 저장
joblib.dump(model, "risk_model.pkl")
joblib.dump(scaler, "risk_scaler.pkl")

print("저장 완료: risk_model.pkl / risk_scaler.pkl")
