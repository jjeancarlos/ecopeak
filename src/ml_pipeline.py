import os
import pandas as pd
import numpy as np
import joblib # Para salvar os modelos
import spacy # Para NLP (NER)

# Importações do Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- 1. Definição de Caminhos ---
PROCESSED_PATH = "data/processed"
MODELS_PATH = "data/models" # <--- CORREÇÃO 1: Caminho alterado

# Garante que a pasta 'models' (dentro de data) exista
os.makedirs(MODELS_PATH, exist_ok=True)

def train_anomaly_model(data_path):
    """
    Treina o modelo Isolation Forest para detecção de anomalias de poluição.
    """
    print("\n--- 1. Treinando Modelo de Detecção de Anomalias (Isolation Forest) ---")
    try:
        df = pd.read_parquet(data_path)
        df = df.dropna()
        
        if df.empty:
            print("Dados de poluição vazios. Pulando treinamento do Isolation Forest.")
            return

        print(f"Dados de poluição carregados: {df.shape}")

        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(df)
        
        save_path = os.path.join(MODELS_PATH, "isolation_forest.joblib")
        joblib.dump(model, save_path)
        print(f"Modelo Isolation Forest salvo em: {save_path}")

    except FileNotFoundError:
        print("Arquivo 'poluicao_processada.parquet' não encontrado.")
    except Exception as e:
        print(f"Erro ao treinar Isolation Forest: {e}")

def create_risk_labels(df):
    """
    Cria a variável alvo (risco) com base no valor da multa.
    """
    df['valor_multa'] = df['valor_multa'].fillna(0)
    
    quantiles = df['valor_multa'][df['valor_multa'] > 0].quantile([0.33, 0.66]).values
    
    def classify_risk(valor):
        if valor <= 0:
            return 'Baixo' 
        elif valor <= quantiles[0]:
            return 'Baixo'
        elif valor <= quantiles[1]:
            return 'Medio'
        else:
            return 'Alto'

    df['risco'] = df['valor_multa'].apply(classify_risk)
    print("Distribuição da variável alvo 'risco':")
    print(df['risco'].value_counts(normalize=True))
    return df

def train_risk_model(data_path):
    """
    Treina o modelo Random Forest para predição de Risco de Autuação.
    """
    print("\n--- 2. Treinando Modelo de Predição de Risco (Random Forest) ---")
    try:
        df = pd.read_parquet(data_path)
        
        # <--- CORREÇÃO 2: Preenche NaNs na coluna de texto ---
        # Isso corrige o erro: 'NoneType' object has no attribute 'lower'
        df['DES_INFRACAO'] = df['DES_INFRACAO'].fillna("")
        # --- FIM DA CORREÇÃO ---
        
        print(f"Dados de autuações carregados: {df.shape}")
        
        # 1. Criar a variável alvo (y)
        df = create_risk_labels(df)
        
        # 2. Definir Features (X) e Target (y)
        FEATURES = [
            'distancia_uc_m', 'tipo_industria', 'mes',            
            'trimestre', 'ano', 'DES_INFRACAO'
        ]
        TARGET = 'risco'
        
        X = df[FEATURES]
        y = df[TARGET]

        # 3. Dividir os dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 4. Criar o Pipeline de Pré-processamento
        numeric_features = ['distancia_uc_m', 'mes', 'trimestre', 'ano']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_features = ['tipo_industria']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        text_feature = 'DES_INFRACAO'
        text_transformer = TfidfVectorizer(
            max_features=1000, 
            stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um']
        )

        # 5. Combinar todos os pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('txt', text_transformer, text_feature)
            ])

        # 6. Criar o Pipeline final (Pré-processador + Modelo)
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))
        ])
        
        # 7. Treinar o modelo
        print("Iniciando treinamento do Random Forest... (Pode demorar um pouco)")
        rf_pipeline.fit(X_train, y_train)
        print("Treinamento concluído.")

        # 8. Avaliar o modelo
        y_pred = rf_pipeline.predict(X_test)
        print("\n--- Métricas de Avaliação (Random Forest) ---")
        print(classification_report(y_test, y_pred))

        # 9. Salvar o pipeline treinado
        save_path = os.path.join(MODELS_PATH, "random_forest_risk_pipeline.joblib")
        joblib.dump(rf_pipeline, save_path)
        print(f"Pipeline de Risco (Random Forest) salvo em: {save_path}")

    except FileNotFoundError:
        print("Arquivo 'autuacoes_processadas.parquet' não encontrado.")
    except Exception as e:
        print(f"Erro ao treinar Random Forest: {e}")

def create_topic_labels(df):
    """
    Cria labels de "temática" baseadas em palavras-chave.
    """
    df['DES_INFRACAO'] = df['DES_INFRACAO'].astype(str).str.lower()
    
    def classify_topic(text):
        if 'fauna' in text or 'pesca' in text or 'animal' in text:
            return 'Fauna/Pesca'
        if 'flora' in text or 'madeira' in text or 'desmatamento' in text or 'vegetal' in text:
            return 'Flora/Desmatamento'
        if 'poluição' in text or 'poluir' in text or 'poluentes' in text:
            return 'Poluicao'
        if 'água' in text or 'rio' in text or 'recursos hídricos' in text:
            return 'Recursos Hidricos'
        return 'Outros'
        
    df['tematica'] = df['DES_INFRACAO'].apply(classify_topic)
    print("\nDistribuição da 'tematica' (NLP):")
    print(df['tematica'].value_counts(normalize=True))
    return df

def train_nlp_topic_model(data_path):
    """
    Treina um modelo de NLP para "Classificação da Temática".
    """
    print("\n--- 3. Treinando Modelo de Classificação de Temática (NLP) ---")
    try:
        df = pd.read_parquet(data_path)
        
        # 1. Criar labels
        df = create_topic_labels(df)
        df_filtered = df[df['tematica'] != 'Outros']

        if df_filtered.empty:
            print("Nenhum dado de NLP para classificar. Pulando.")
            return

        # 2. Definir X e y
        X = df_filtered['DES_INFRACAO']
        y = df_filtered['tematica']

        # 3. Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 4. Criar pipeline de NLP
        nlp_pipeline = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um'])),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=50))
        ])

        # 5. Treinar
        print("Iniciando treinamento do modelo de Temática (NLP)...")
        nlp_pipeline.fit(X_train, y_train)
        print("Treinamento concluído.")

        # 6. Avaliar
        y_pred = nlp_pipeline.predict(X_test)
        print("\n--- Métricas de Avaliação (Classificação de Temática NLP) ---")
        print(classification_report(y_test, y_pred))

        # 7. Salvar
        save_path = os.path.join(MODELS_PATH, "nlp_topic_pipeline.joblib")
        joblib.dump(nlp_pipeline, save_path)
        print(f"Pipeline de Temática (NLP) salvo em: {save_path}")

    except FileNotFoundError:
        print("Arquivo 'autuacoes_processadas.parquet' não encontrado.")
    except Exception as e:
        print(f"Erro ao treinar modelo de tópicos NLP: {e}")

def test_ner_model():
    """
    Carrega o spaCy para Extração de Entidades (NER) - Não treina, apenas demonstra.
    """
    print("\n--- 4. Testando Extração de Entidades (NER com spaCy) ---")
    try:
        nlp = spacy.load("pt_core_news_lg")
        exemplo = "A empresa Madeireira Taperu Ltda foi multada por desmatamento ilegal na Amazônia, perto de Manaus."
        
        doc = nlp(exemplo)
        
        print(f"Texto de exemplo: {exemplo}")
        print("Entidades Extraídas:")
        for ent in doc.ents:
            print(f"  - {ent.text} ({ent.label_})")
        
        print("Modelo NER (spaCy) carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar modelo spaCy 'pt_core_news_lg': {e}")
        print("Certifique-se de que você rodou: python -m spacy download pt_core_news_lg")

def main():
    """Orquestrador do pipeline de treinamento de ML."""
    print("====== INICIANDO PIPELINE DE MACHINE LEARNING ======")
    
    # Modelo 1: Detecção de Anomalias de Poluição
    train_anomaly_model(os.path.join(PROCESSED_PATH, "poluicao_processada.parquet"))
    
    # Modelo 2: Predição de Risco (Severidade)
    train_risk_model(os.path.join(PROCESSED_PATH, "autuacoes_processadas.parquet"))
    
    # Modelo 3: Classificação de Temática (NLP)
    train_nlp_topic_model(os.path.join(PROCESSED_PATH, "autuacoes_processadas.parquet"))
    
    # Modelo 4: Teste de Extração de Entidades (NER)
    test_ner_model()
    
    print("\n====== PIPELINE DE MACHINE LEARNING CONCLUÍDO ======")
    print(f"Modelos salvos em: {MODELS_PATH}/")

if __name__ == "__main__":
    main()