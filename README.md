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
