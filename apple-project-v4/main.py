import joblib
import pandas as pd
import numpy as np
import sqlite3
import datetime
import os
import socket  # <--- Importante para o teste de conexão
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.api_core import exceptions as google_exceptions # Para capturar erros específicos do Google

# --- Importa e carrega o .env ---
from dotenv import load_dotenv
load_dotenv() 

app = FastAPI(title="AppleSmart API v3 (Gemini Blindado) 🍎", version="3.2")

# Configuração de CORS (Permite acesso de qualquer origem)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- 1. CONFIGURAÇÃO GERAL ---
CHAVE_API = os.environ.get("GEMINI_API_KEY")

if CHAVE_API:
    genai.configure(api_key=CHAVE_API)
    print("✅ Gemini AI configurado!")
else:
    print("⚠️ AVISO: Chave GEMINI_API_KEY não encontrada no .env!")

# --- 2. BANCO DE DADOS (SQLite) ---
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

# --- 3. CARREGAMENTO DOS MODELOS ML ---
pesos = None
scaler = None

# Funções de ativação manuais (NumPy puro)
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

@app.on_event("startup")
def load_artifacts():
    global pesos, scaler
    try:
        # Certifique-se que esses arquivos estão na mesma pasta
        pesos = joblib.load("modelo_apple_lite.pkl")
        scaler = joblib.load("scaler_apple_vfinal.pkl")
        print("✅ Modelos de ML carregados com sucesso!")
    except Exception as e:
        print(f"❌ CRÍTICO: Erro ao carregar modelos .pkl: {e}")

# --- 4. FUNÇÕES AUXILIARES DE CONEXÃO E IA ---

def checar_conexao():
    """
    Tenta conectar ao DNS do Google (8.8.8.8) na porta 53.
    Timeout super curto de 1.5s. Se falhar, assume que está sem internet.
    Isso evita que a biblioteca do Gemini tente conectar e fique travada.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=1.5)
        return True
    except OSError:
        return False

def gerar_receita_gemini(doçura, acidez, textura, suculencia):
    # 1. Validação Básica
    if not CHAVE_API:
        return "⚠️ Erro: API Key não configurada."

    # 2. O PULO DO GATO: Teste de Conexão Rápido
    # Se isso falhar, retorna IMEDIATAMENTE. Não deixa a API travar.
    print("📡 Testando conexão com a internet...")
    if not checar_conexao():
        print("❌ Sem internet detectada no teste de ping.")
        return "⚠️ Modo Offline: Sem conexão para gerar receita."

    # 3. Chamada à IA com Timeout de Segurança
    try:
        print("⏳ Internet OK. Chamando Gemini...")
        model = genai.GenerativeModel('gemini-flash-latest')

        prompt = f"""
        Atue como um Chef Sustentável industrial. Recebemos uma maçã imprópria para venda in natura, mas comestível.
        Dados sensoriais: 
        - Doçura: {doçura:.2f} (Normalizado)
        - Acidez: {acidez:.2f}
        - Crocância: {textura:.2f}
        - Suculência: {suculencia:.2f}

        Sugira UMA única receita criativa, viável industrialmente e curta (máximo 15 palavras) para evitar desperdício.
        Exemplo: "Geleia rústica de maçã com pimenta rosa."
        """
        
        # 'request_options' impõe um limite de x segundos para o Google responder
        response = model.generate_content(prompt, request_options={'timeout': 8})
        return response.text.strip()

    except google_exceptions.DeadlineExceeded:
        return "⚠️ Tempo limite excedido (Internet lenta)."
    except google_exceptions.ServiceUnavailable:
        return "⚠️ Serviço Gemini indisponível temporariamente."
    except Exception as e:
        print(f"⚠️ Erro genérico na IA: {e}")
        return "⚠️ Erro na IA (Sistema operando em contingência)."

# --- 5. ENDPOINTS DA API ---

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
    # A. Prepara os dados (Engenharia de Features)
    dados = maca.dict()
    df = pd.DataFrame([dados])
    
    # Recria as features calculadas no treinamento
    df['Flavor_Score'] = df['Sweetness'] + df['Juiciness']
    df['Texture_Score'] = df['Crunchiness'] - df['Ripeness']
    
    # Garante a ordem exata das colunas
    cols = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 
            'Ripeness', 'Acidity', 'Flavor_Score', 'Texture_Score']
    
    try:
        X = scaler.transform(df[cols])
    except Exception as e:
        return {"erro": f"Erro no preprocessamento (Scaler): {str(e)}"}

    # B. Inferência Local (Matemática Pura - Funciona Offline)
    # Camada 1
    layer1 = relu(np.dot(X, pesos['W1']) + pesos['b1'])
    # Camada 2
    layer2 = relu(np.dot(layer1, pesos['W2']) + pesos['b2'])
    # Saída
    output = sigmoid(np.dot(layer2, pesos['W3']) + pesos['b3'])
    
    prob = float(output[0][0])
    
    # Limiar calibrado no TCC: 0.53
    veredito = "APROVADA 🍎" if prob > 0.53 else "REPROVADA 🤢"

    # C. Lógica Híbrida (IA Generativa apenas se Reprovada)
    receita = "N/A - Venda In Natura"

    if veredito == "REPROVADA 🤢":
        # Chama a função blindada
        receita = gerar_receita_gemini(maca.Sweetness, maca.Acidity, maca.Crunchiness, maca.Juiciness)

    # D. Auditoria (Log no SQLite)
    try:
        conn = sqlite3.connect('historico.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO auditoria (data_hora, tamanho, doçura, resultado, probabilidade, receita_sugerida)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.datetime.now(), maca.Size, maca.Sweetness, veredito, prob, receita))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao salvar no banco (não crítico): {e}")

    # E. Resposta Final JSON
    return {
        "veredito": veredito,
        "probabilidade": round(prob, 4),
        "sugestao_gemini": receita,
        "status_conexao": "Online" if receita not in ["⚠️ Modo Offline: Sem conexão para gerar receita."] else "Offline"
    }

@app.get("/historico")
def ler_historico():
    try:
        conn = sqlite3.connect('historico.db')
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM auditoria ORDER BY id DESC LIMIT 20")
        dados = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return dados
    except Exception as e:
        return {"erro": str(e)}

# Para rodar: uvicorn main:app --reload