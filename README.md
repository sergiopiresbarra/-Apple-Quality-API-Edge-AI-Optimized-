# 🍎 Apple Quality API (Edge AI Optimized)

Este projeto é uma API de Inteligência Artificial para classificar a qualidade de maçãs com base em características físico-químicas.

O diferencial deste projeto é a implementação de **Inferência NumPy-Only**. 
Para contornar limitações de memória em ambientes de produção restritos (como o plano gratuito do Replit ou dispositivos IoT), a dependência do framework `TensorFlow` foi removida da etapa de inferência. A "forward pass" da rede neural foi reescrita utilizando apenas álgebra linear com `NumPy`, reduzindo o tamanho da imagem Docker em **~500MB** e o uso de RAM drasticamente.

## 🛠️ Tecnologias

* **Python 3.9+**
* **FastAPI:** Framework moderno e assíncrono para a API.
* **Scikit-Learn:** Para pré-processamento (StandardScaler).
* **NumPy:** Para cálculos matriciais da rede neural.
* **Docker:** Para conteinerização e deploy.

## 🧠 Engenharia de Features & Modelo

O modelo original foi treinado com Keras/TensorFlow utilizando um dataset de características de maçãs (Tamanho, Peso, Doçura, etc.).
Durante o pipeline, duas novas features são calculadas em tempo real:
* `Flavor_Score` = Sweetness + Juiciness
* `Texture_Score` = Crunchiness - Ripeness

## 🚀 Como Rodar Localmente

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU-USUARIO/apple-quality-api.git](https://github.com/SEU-USUARIO/apple-quality-api.git)
   cd apple-quality-api

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt

3. **Inicie o Servidor:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000

4. **Teste:**
   Acesse http://localhost:8000/docs para usar a interface interativa (Swagger UI).

## ⚡ Otimização (TensorFlow-Free)

O modelo treinado (.keras) foi decomposto, extraindo-se os pesos (Weights) e viéses (Biases) de cada camada densa. A inferência é realizada através da multiplicação de matrizes manual:
   ```python
      # Exemplo da lógica implementada (sem TensorFlow)
      layer1 = relu(np.dot(X, W1) + b1)
      layer2 = relu(np.dot(layer1, W2) + b2)
      output = sigmoid(np.dot(layer2, W3) + b3)
   ```

Isso garante que o modelo rode em qualquer ambiente com suporte mínimo a Python, sem necessidade de instalar bibliotecas pesadas de Deep Learning.

## 🌟 Novas Features (v2.0)

### 1. Integração com IA Generativa (GenAI)
O sistema não apenas classifica, mas propõe soluções de negócio.
- **Fluxo:** Se uma maçã é reprovada (`Prob < 0.53`), o sistema aciona a API do **Google Gemini (LLM)** via *Prompt Engineering*.
- **Resultado:** A IA analisa as falhas (ex: excesso de acidez) e sugere uma receita culinária personalizada (ex: "Membrillo de Maçã Rústico") para recuperar o valor do produto que seria descartado.

### 2. Persistência de Dados (SQL)
Implementação de banco de dados relacional (SQLite) para rastreabilidade.
- Todo teste realizado é logado na tabela `auditoria` com timestamp, métricas de entrada, veredito da IA Clássica e sugestão da IA Generativa.

## 🌟 Novas Features (v3.0)

**Interface Gráfica (Front-end):** Desenvolvimento de um Dashboard em HTML/JS para facilitar o uso por usuários não técnicos.
**Auditoria Visual:** Nova rota `/historico` conectada ao Front-end, permitindo visualizar as últimas análises e decisões da IA em tempo real.
