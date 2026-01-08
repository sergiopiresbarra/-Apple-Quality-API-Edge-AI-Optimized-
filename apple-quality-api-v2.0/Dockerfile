# 1. Imagem Base: Começamos com um Python leve (Slim) para economizar espaço
FROM python:3.9-slim

# 2. Pasta de Trabalho: Onde vamos colocar as coisas dentro do container
WORKDIR /app

# 3. Copiar Dependências: Primeiro copiamos só a lista de compras (Cache eficiente)
COPY requirements.txt .

# 4. Instalar Dependências: Instalamos tudo
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o Código: Agora trazemos o resto (main.py, .pkl, .db)
COPY . .

# 6. Expor a Porta: Avisamos que a API roda na porta 8000
EXPOSE 8000

# 7. Comando Final: O que roda quando o container liga
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]