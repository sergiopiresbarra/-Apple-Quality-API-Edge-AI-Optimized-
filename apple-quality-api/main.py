import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Apple Quality API (Numpy Edition) ⚡", version="2.0")

# Variáveis globais
pesos = None
scaler = None

# FUNÇÃO DE ATIVAÇÃO (A matemática que o TF fazia)
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@app.on_event("startup")
def load_artifacts():
    global pesos, scaler
    try:
        # Carregamos o arquivo .pkl leve em vez do .keras pesado
        pesos = joblib.load("modelo_apple_lite.pkl")
        scaler = joblib.load("scaler_apple_vfinal.pkl")
        print("✅ MODELO ULTRA-LEVE CARREGADO!")
    except Exception as e:
        print(f"❌ Erro: {e}")

class MacaInput(BaseModel):
    Size: float
    Weight: float
    Sweetness: float
    Crunchiness: float
    Juiciness: float
    Ripeness: float
    Acidity: float

@app.post("/predict")
def predict_apple(maca: MacaInput):
    # 1. Preparar Dados (Igual antes)
    dados = maca.dict()
    df = pd.DataFrame([dados])
    df['Flavor_Score'] = df['Sweetness'] + df['Juiciness']
    df['Texture_Score'] = df['Crunchiness'] - df['Ripeness']

    cols = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 
            'Ripeness', 'Acidity', 'Flavor_Score', 'Texture_Score']

    # 2. Escalonar
    X = scaler.transform(df[cols])

    # 3. INFERÊNCIA MANUAL (O Segredo 🤫)
    # Reproduzimos o caminho da rede neural na mão: Entrada -> Camada 1 -> Camada 2 -> Saída

    # Camada 1: X * W1 + b1 (com ativação ReLU)
    layer1 = relu(np.dot(X, pesos['W1']) + pesos['b1'])

    # Camada 2: layer1 * W2 + b2 (com ativação ReLU)
    layer2 = relu(np.dot(layer1, pesos['W2']) + pesos['b2'])

    # Saída: layer2 * W3 + b3 (com ativação Sigmoid)
    output = sigmoid(np.dot(layer2, pesos['W3']) + pesos['b3'])

    prob = float(output[0][0])

    return {
        "veredito": "APROVADA 🍎" if prob > 0.53 else "REPROVADA 🤢",
        "probabilidade": round(prob, 4),
        "metodo": "Inferência Numpy-Only (Sem TensorFlow)"
    }