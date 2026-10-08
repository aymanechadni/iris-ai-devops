
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# ==============================================
# 1. Creation de l'application FastAPI
# ==============================================

app = FastAPI(
    title="Mini Projet IA - DevOps",
    description="Classification des fleurs Iris avec Machine Learning",
    version="1.0.0"
)

# ==============================================
# 2. Chargement du dataset Iris
# ==============================================

iris = load_iris()

X = iris.data
y = iris.target

# ==============================================
# 3. Entrainement du modele Machine Learning
# ==============================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# ==============================================
# 4. Definition des donnees d'entree
# ==============================================

class Flower(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

# ==============================================
# 5. Route d'accueil
# ==============================================

@app.get("/")
def home():
    return {
        "message": "Bienvenue dans notre application IA DevOps",
        "status": "running",
        "model": "Random Forest",
        "version": "1.0.0"
    }

# ==============================================
# 6. Route de verification
# ==============================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# ==============================================
# 7. Route de prediction IA
# ==============================================

@app.post("/predict")
def predict(flower: Flower):

    features = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    flower_name = str(iris.target_names[prediction])

    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction])

    return {
        "prediction": flower_name,
        "confidence": round(confidence, 4),
        "model": "Random Forest"
    }
