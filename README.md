# 🌍 EcoPeak: Sistema de Monitoramento Ambiental Industrial

Um sistema de análise de dados e Machine Learning para prever violações ambientais e identificar geograficamente áreas de risco. O projeto utiliza dados públicos de autuações do IBAMA e dados geoespaciais de Unidades de Conservação (UCs) do MMA para treinar um modelo de previsão de risco (Alto, Médio, Baixo).

## 🎯 Objetivo Principal

O objetivo deste sistema é analisar dados históricos para **prever violações ambientais futuras e identificar áreas geográficas de risco**, permitindo que as agências de fiscalização otimizem a alocação de recursos e atuem de forma mais proativa.

## ✨ Principais Features

  * **Dashboard Interativo (Streamlit):** Uma interface web para visualizar:
      * O mapa de autuações classificado por risco (Alto, Médio, Baixo).
      * Alertas de anomalia de poluição em "tempo real".
      * Gráficos de tendências de infrações ao longo dos anos.
  * **Modelo de Risco (Random Forest):** Prevê o nível de risco (Alto, Médio, Baixo) de uma nova infração com base em dados históricos, alcançando **61% de acurácia**.
  * **Classificação NLP (Random Forest + TF-IDF):** Classifica automaticamente o tema de uma infração (ex: Flora, Fauna, Poluição) a partir do texto da autuação, com **100% de precisão** (baseado na metodologia de validação).
  * **Engenharia Geoespacial (Geopandas):** O sistema calcula dinamicamente a distância de cada infração até a Unidade de Conservação (UC) mais próxima. Esta feature (`distancia_uc_m`) foi identificada como o **fator preditivo mais importante (peso de 61%)** para determinar o risco.

## 💻 Tech Stack

  * **Linguagem:** Python 3.12
  * **Análise de Dados:** Pandas, Geopandas
  * **Machine Learning:** Scikit-learn, Spacy
  * **Dashboard:** Streamlit
  * **Visualização:** Plotly, Seaborn, Matplotlib
  * **Análise Exploratória:** JupyterLab

-----

## 📁 Estrutura do Projeto

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
│   └── dashboard.py         # Dashboard (Dash)
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

## 🚀 Instalação e Execução

Siga os passos abaixo para configurar e executar o projeto localmente.

### 1\. Pré-requisitos

  * Python 3.10+
  * A instalação da biblioteca **Geopandas** pode exigir dependências de sistema adicionais (como `libgdal` ou `shapely`). Recomenda-se consultar a [documentação oficial do Geopandas](https://www.google.com/search?q=https://geopandas.org/en/stable/installation.html) para instruções específicas do seu sistema operacional.

### 2\. Clonar o Repositório

```bash
git clone https://github.com/jjeancarlos/ecopeak.git
cd ecopeak/
```

### 3\. Configurar o Ambiente Virtual

```bash
# Criar o ambiente
python3 -m venv .venv

# Ativar o ambiente (Linux/macOS)
source .venv/bin/activate
# ou (Windows)
# .\.venv\Scripts\activate
```

### 4\. Instalar as Dependências

```bash
pip install -r requirements.txt
```

### 5\. Baixar o Modelo de Linguagem (spaCy)

```bash
python -m spacy download pt_core_news_lg
```

-----

## ⚙️ Executando o Pipeline Completo

O pipeline deve ser executado na ordem correta para coletar os dados, processá-los e treinar os modelos.

### 1\. Coleta de Dados

Coleta os dados brutos do IBAMA e MMA e salva em `data/raw/`.

```bash
python src/data_collection.py
```

### 2\. Processamento e Engenharia de Features

Limpa os dados, calcula a distância até as UCs e salva os datasets processados em `data/processed/`.

```bash
python src/data_processing.py
```

### 3\. Treinamento dos Modelos de ML

Treina os modelos de Risco, NLP e Anomalia. Salva os artefatos (`.joblib`) em `data/models/`.

```bash
python src/ml_pipeline.py
```

### 4\. Iniciar o Dashboard

Inicia a aplicação web do Streamlit.

```bash
streamlit run src/dashboard.py
```

Acesse [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501) no seu navegador.

-----

## 🔬 Análise e Avaliação (Notebooks)

Para uma análise exploratória (EDA) interativa ou para ver a avaliação detalhada dos modelos (Matriz de Confusão, Feature Importance), utilize os notebooks.

```bash
# Inicia o servidor do Jupyter (na pasta ecopeak/)
jupyter lab
```

Acesse os arquivos `notebooks/eda.ipynb` e `notebooks/model_evaluation.ipynb` na interface do Jupyter.

## 📊 Principais Resultados da Análise

  * **Acurácia do Risco:** O modelo de Random Forest alcançou **61% de acurácia** na previsão do nível de risco.
  * **Principal Fator de Risco:** A distância até uma Unidade de Conservação (`distancia_uc_m`) é o fator preditivo mais importante, com um **peso de 61%** no modelo.
  * **Análise de Tendência:** A análise histórica (desde 2005) mostra um pico claro de infrações relacionadas à "Flora/Desmatamento" entre 2014 e 2018.
