# 🌍 **EcoPeak: Sistema de Monitoramento Ambiental Industrial**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Ativo-success)

> 🔎 **Uma plataforma inteligente para prever riscos ambientais, analisar infrações e visualizar áreas de vulnerabilidade ecológica no Brasil.**

O **EcoPeak** combina análise de dados, Machine Learning e geoprocessamento para prever violações ambientais com base em autuações do IBAMA e em Unidades de Conservação do MMA. A plataforma oferece análises preditivas, classificação automática de temas e georreferenciamento de infrações ambientais.

---

# 🎯 **Objetivo Principal**

Analisar dados históricos de infrações ambientais e **prever o nível de risco (Alto, Médio, Baixo)** de futuras ocorrências, além de **mapear geograficamente áreas vulneráveis** para apoiar a tomada de decisão e a fiscalização ambiental.

---

# ✨ **Principais Features**

### 📊 **Dashboard Interativo (Streamlit)**

* Mapa de infrações com classificação de Risco
* Gráficos temporais e tendências
* Sinalização de anomalias ambientais
* Visualização dos resultados do modelo

### 🤖 **Modelo de Risco (Random Forest)**

* Predição do nível de risco futuro
* Acurácia: **61%**

### 📝 **Classificação NLP (TF-IDF + Random Forest)**

* Classifica automaticamente o tema da infração
* Precisão avaliada em **100%** (amostragem validada)

### 🗺️ **Engenharia Geoespacial (Geopandas)**

* Distância automática até a Unidade de Conservação mais próxima
* `distancia_uc_m` = **feature mais importante (61% de importância)**

---

# 💻 **Tech Stack**

| Categoria            | Ferramentas                 |
| -------------------- | --------------------------- |
| Linguagem            | Python 3.12                 |
| Dados                | Pandas, Geopandas           |
| Machine Learning     | scikit-learn, spaCy         |
| Visualização         | Plotly, Matplotlib, Seaborn |
| Dashboard            | Streamlit                   |
| Análise Exploratória | Jupyter Lab                 |



# 📁 **Estrutura do Projeto**

```text
projeto/
├── data/
│   ├── raw/                 # Dados brutos das APIs
│   ├── processed/           # Dados processados
│   └── models/              # Modelos treinados
│
├── src/
│   ├── data_collection.py   # Coleta de APIs
│   ├── data_processing.py   # Processamento
│   ├── ml_pipeline.py       # Pipeline ML
│   └── dashboard.py         # Dashboard (Streamlit)
│
├── notebooks/
│   ├── eda.ipynb            # Análise exploratória
│   └── model_evaluation.ipynb
│
├── reports/
│   └── analise_preditiva.pdf
│
└── requirements.txt
```

---

# 🚀 **Instalação e Execução**

## 1️⃣ Pré-requisitos

* Python 3.10+
* Dependências geoespaciais para o Geopandas (GDAL, Shapely)
* Recomendado: ambiente virtual

---

## 2️⃣ Clonar o Repositório

```bash
git clone https://github.com/jjeancarlos/ecopeak.git
cd ecopeak/
```

---

## 3️⃣ Configurar o Ambiente Virtual

### Criar ambiente

```bash
python3 -m venv .venv
```

### Ativar (Linux/Mac)

```bash
source .venv/bin/activate
```

### Ativar (Windows)

```bash
.\.venv\Scripts\activate
```

---

## 4️⃣ Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## 5️⃣ Instalar o Modelo Linguístico do spaCy

O spaCy **não instala modelos via pip**, então execute:

```bash
python -m spacy download pt_core_news_lg
```

Caso queira um modelo mais leve:

```bash
python -m spacy download pt_core_news_sm
```

---

# 🔄 **Executando o Pipeline Completo**

## 1. Coleta de Dados

```bash
python src/data_collection.py
```

## 2. Processamento + Engenharia de Features

```bash
python src/data_processing.py
```

## 3. Treinamento dos Modelos

```bash
python src/ml_pipeline.py
```

## 4. Executar o Dashboard

```bash
streamlit run src/dashboard.py
```

Acesse em:
👉 [http://localhost:8501](http://localhost:8501)

---

# 📓 **Análises e Notebooks**

Execute:

```bash
jupyter lab
```

Abra:

* `notebooks/eda.ipynb`
* `notebooks/model_evaluation.ipynb`

---

# 📊 **Principais Resultados**

* **Acurácia geral (Random Forest Risco):** 61%
* **Feature mais importante:** distância até UC (61% do peso)
* **Tendência histórica:** picos de desmatamento entre 2014–2018
* **NLP:** classificação com 100% na validação interna

---

# 🛣️ **Roadmap**

* [ ] Otimização da Random Forest
* [ ] Adicionar algoritmo de Explainable AI (SHAP)
* [ ] Criar API REST com FastAPI
* [ ] Adicionar autenticação para o dashboard
* [ ] Criar versão containerizada (Docker)

---

# 🤝 **Como Contribuir**

1. Faça um fork
2. Crie uma branch (`feature/nova-feature`)
3. Commit suas mudanças
4. Abra um Pull Request

---

# 📜 **Licença**

Este projeto está licenciado sob a **MIT License**.
Veja o arquivo `LICENSE` para mais detalhes.