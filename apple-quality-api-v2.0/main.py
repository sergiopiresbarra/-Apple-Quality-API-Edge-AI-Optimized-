import joblib
import pandas as pd
import numpy as np
import sqlite3
import datetime
import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AppleSmart API v3 (Gemini Real) 🍎", version="3.0")

# --- 1. CONFIGURAÇÃO DO GEMINI ---
# Pega a chave segura do Replit
CHAVE_API = os.environ.get("GEMINI_API_KEY")

if CHAVE_API:
    genai.configure(api_key=CHAVE_API)
    print("✅ Gemini AI conectado!")
else:
    print("⚠️ AVISO: Chave GEMINI_API_KEY não encontrada nas Secrets!")

# --- 2. BANCO DE DADOS E MODELO (IGUAL ANTES) ---
def init_db():
    conn = sqlite3.connect('historico.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS auditoria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            tamanho REAL,
            doçura REAL,
            resultado TEXT,
            probabilidade REAL,
            receita_sugerida TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

pesos = None
scaler = None
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

@app.on_event("startup")
def load_artifacts():
    global pesos, scaler
    try:
        pesos = joblib.load("modelo_apple_lite.pkl")
        scaler = joblib.load("scaler_apple_vfinal.pkl")
        print("✅ Modelos carregados!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")

# --- 3. FUNÇÃO COM IA REAL (GEMINI) ---
def gerar_receita_gemini(doçura, acidez, textura, suculencia):
    if not CHAVE_API:
        return "Erro: API Key não configurada."

    try:
        # Usamos o alias genérico que apareceu na sua lista
        model = genai.GenerativeModel('gemini-flash-latest')

        prompt = f"""
        Atue como um Chef Sustentável. Recebemos uma maçã imprópria para venda (feia ou pequena), mas comestível.
        Características:
        - Doçura: {doçura:.2f} (Escala padronizada)
        - Acidez: {acidez:.2f}
        - Crocância: {textura:.2f}
        - Suculência: {suculencia:.2f}

        Baseado nisso, sugira UMA receita criativa e curta (máximo 1 frase) para industrializar essa fruta e evitar desperdício.
        Exemplo: "Compota picante de maçã com canela."
        """

        response = model.generate_content(prompt)
        return response.text.strip() # Limpa espaços extras

    except Exception as e:
        return f"Erro na IA: {str(e)}"

# --- 4. ENDPOINTS ---
class MacaInput(BaseModel):
    Size: float
    Weight: float
    Sweetness: float
    Crunchiness: float
    Juiciness: float
    Ripeness: float
    Acidity: float

@app.post("/predict_and_genai")
def predict_apple(maca: MacaInput):
    # a. Prepara dados
    dados = maca.dict()
    df = pd.DataFrame([dados])
    df['Flavor_Score'] = df['Sweetness'] + df['Juiciness']
    df['Texture_Score'] = df['Crunchiness'] - df['Ripeness']
    cols = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 
            'Ripeness', 'Acidity', 'Flavor_Score', 'Texture_Score']
    X = scaler.transform(df[cols])

    # b. Inferência Local (NumPy)
    layer1 = relu(np.dot(X, pesos['W1']) + pesos['b1'])
    layer2 = relu(np.dot(layer1, pesos['W2']) + pesos['b2'])
    output = sigmoid(np.dot(layer2, pesos['W3']) + pesos['b3'])
    prob = float(output[0][0])

    veredito = "APROVADA 🍎" if prob > 0.53 else "REPROVADA 🤢"

    # c. CHAMADA PARA O GEMINI (Só se reprovada)
    receita = "N/A - Venda In Natura"

    if veredito == "REPROVADA 🤢":
        print("⏳ Consultando o Gemini...")
        receita = gerar_receita_gemini(maca.Sweetness, maca.Acidity, maca.Crunchiness, maca.Juiciness)

    # d. Salva no SQL
    conn = sqlite3.connect('historico.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO auditoria (data_hora, tamanho, doçura, resultado, probabilidade, receita_sugerida)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.datetime.now(), maca.Size, maca.Sweetness, veredito, prob, receita))
    conn.commit()
    conn.close()

    return {
        "veredito": veredito,
        "probabilidade": round(prob, 4),
        "sugestao_gemini": receita
    }
