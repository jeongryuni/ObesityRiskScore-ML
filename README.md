# SaveUs — 비만 위험도 모델 v2 (Obesity Risk Score)

본 문서는 SaveUs 서비스에서 사용되는 **새로운 비만 위험도(Obesity Risk Score) 모델**의 공식 기술 문서입니다.  
이번 버전은 “오늘 먹은 식단 기반 위험도 0~100점 계산”을 핵심 목표로 설계되었습니다.

---

# 1. 모델 목표

- 식단을 많이 먹으면 위험도가 올라가는 **직관적 구조**
- 결과는 **0~100 사이 연속값**
- 의료적 기준을 반영한 **명확한 가중치 기반 점수 공식**
- 식단 데이터가 즉시 반영되는 **실시간 분석 모델**

---

# 2. 위험도 점수 산출 공식 (Risk Score Formula)

비만 위험도는 다음 네 가지 영양 성분을 기준으로 계산합니다:

```text
risk_score =
    (total_calories / 2500 * 30) +
    (total_fat / 70 * 25) +
    (total_sugar / 50 * 20) +
    (total_sodium / 2000 * 25)
```

---

# 3. 공식 설명

각 영양 성분별 기여도는 아래와 같이 설계했습니다.

| 항목 | 기준량 | 가중치(점수) | 설명 |
|------|---------|---------------|-------|
| total_calories | 2500 kcal | 30점 | 가장 중요한 요소 |
| total_fat | 70 g | 25점 | 지방 섭취 증가 시 위험 증가 |
| total_sugar | 50 g | 20점 | 당류 고섭취 시 대사질환 위험 증가 |
| total_sodium | 2000 mg | 25점 | 나트륨 과다 섭취 반영 |

총합 = **100점 만점**,  
따라서 사용자 위험도는 항상 **0~100 사이 자연스러운 연속형 값**이 됩니다.

---

# 4. 학습 데이터 구성

### 입력(X)
- total_calories  
- total_fat  
- total_sugar  
- total_sodium  
- carb_ratio  
- protein_ratio  

### 출력(y)
- 상단 공식으로 계산된 **risk_score (0~100)**

---

# 5. 머신러닝 모델

본 모델은 **RandomForestRegressor 기반 연속형 모델**입니다.

선정 이유:
- 연속값 예측(0~100)에 적합  
- 적은 학습 데이터에서도 안정적  
- 과적합을 자동으로 완화  
- 영양소 간 상호작용 반영

---

# 6. 학습 코드 (train.py)

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load nutrition dataset
df = pd.read_csv("nutrition.csv")

# Features
X = df[[
    "total_calories",
    "total_fat",
    "total_sugar",
    "total_sodium",
    "carb_ratio",
    "protein_ratio"
]]

# Risk Score 생성
df["risk_score"] = (
    (df["total_calories"] / 2500 * 30) +
    (df["total_fat"] / 70 * 25) +
    (df["total_sugar"] / 50 * 20) +
    (df["total_sodium"] / 2000 * 25)
)

y = df["risk_score"]

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "obesity_model_v2.pkl")

print("Model training complete: obesity_model_v2.pkl saved.")
```

---

# 7. FastAPI 예측 API

```python
@app.get("/predict-risk/{user_id}")
def predict_risk(user_id: int):
    data = get_today_nutrition(user_id)

    features = [[
        data["total_calories"],
        data["total_fat"],
        data["total_sugar"],
        data["total_sodium"],
        data["carb_ratio"],
        data["protein_ratio"],
    ]]

    pred = model.predict(features)[0]
    pred = max(0, min(100, pred))

    return {
        "user_id": user_id,
        "risk_score": round(pred, 2)
    }
```

---

# 8. Spring 연동 코드

```java
public int getObesityPercent(int userId) {
    String url = "http://3.37.90.119:8001/predict-risk/" + userId;
    Map<String, Object> result = restTemplate.getForEntity(url, Map.class).getBody();

    double score = 0.0;
    if (result != null && result.get("risk_score") != null) {
        score = Double.parseDouble(result.get("risk_score").toString());
    }

    return (int) Math.round(score);
}
```

---

# 9. 장점 요약

- BMI 0% 수준 반영 (영향 제거)
- “오늘 식단” 기반으로 실시간 계산  
- 0~100 자연스러운 연속형 점수  
- 의료 기준 기반 가중치 적용  
- 설명 가능한 구조  
- FastAPI → Spring → 화면까지 연동 쉬움  

---

# 10. 결론

이 모델은 SaveUs의 비만 위험도 분석 표준 모델입니다.  
사용자가 오늘 먹은 음식만으로 **즉시 계산되는 건강 위험도**를 제공합니다.

