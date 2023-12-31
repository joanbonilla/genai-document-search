FROM python:3.10-slim

WORKDIR /app

COPY modules/* ./
COPY Pipfile Pipfile.lock ./

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install --dev --system --deploy

COPY . .

EXPOSE 8080

ENTRYPOINT streamlit run app.py --server.address=0.0.0.0 --server.port=8080
