import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# --- 1. Configuração da Página ---
st.set_page_config(
    page_title="Monitoramento Ambiental",
    page_icon="🌍",
    layout="wide"
)

# --- 2. Definição de Caminhos ---
PROCESSED_PATH = "data/processed"
MODELS_PATH = "data/models"

# --- 3. Funções de Carregamento (com Cache) ---
@st.cache_data
def load_data(file_path):
    """Carrega dados .parquet."""
    if not os.path.exists(file_path):
        st.error(f"Erro: Arquivo não encontrado em {file_path}")
        return None
    return pd.read_parquet(file_path)

@st.cache_resource
def load_model(model_path):
    """Carrega modelos .joblib."""
    if not os.path.exists(model_path):
        st.error(f"Erro: Modelo não encontrado em {model_path}")
        return None
    return joblib.load(model_path)

# --- 4. Carregamento dos Dados e Modelos ---
with st.spinner('Carregando dados e modelos...'):
    df_autuacoes = load_data(os.path.join(PROCESSED_PATH, "autuacoes_processadas.parquet"))
    df_poluicao = load_data(os.path.join(PROCESSED_PATH, "poluicao_processada.parquet"))
    
    model_risco = load_model(os.path.join(MODELS_PATH, "random_forest_risk_pipeline.joblib"))
    model_anomalia = load_model(os.path.join(MODELS_PATH, "isolation_forest.joblib"))
    model_tema = load_model(os.path.join(MODELS_PATH, "nlp_topic_pipeline.joblib"))

if df_autuacoes is None or model_risco is None:
    st.error("Falha ao carregar dados essenciais ou modelo de risco. Verifique os caminhos.")
    st.stop()

# --- 5. Título do Dashboard ---
st.title("🌍 Sistema de Monitoramento Ambiental Industrial")

# --- 6. Geração de Predições ---
@st.cache_data
def get_predictions(df, _model): # <--- CORREÇÃO 1: Adicionado '_' ao nome do argumento
    """Gera predições de risco para o dataframe de autuações."""
    df_predict = df.copy()
    df_predict['DES_INFRACAO'] = df_predict['DES_INFRACAO'].fillna("")
    
    features = [
        'distancia_uc_m', 'tipo_industria', 'mes',            
        'trimestre', 'ano', 'DES_INFRACAO'
    ]
    
    # Garante que todas as colunas existem
    features_presentes = [col for col in features if col in df_predict.columns]
    df_predict = df_predict[features_presentes]
    
    # Gera predições
    # <--- CORREÇÃO 2: Usando '_model' para prever
    df['risco_predito'] = _model.predict(df_predict)
    
    # Mapeia cores para o risco
    color_map = {
        'Alto': [255, 0, 0],  # Vermelho
        'Medio': [255, 165, 0], # Laranja
        'Baixo': [0, 128, 0]   # Verde
    }
    df['cor'] = df['risco_predito'].map(color_map)
    return df

# Gera predições de risco para o mapa
df_autuacoes = get_predictions(df_autuacoes, model_risco)

# --- 7. Layout do Dashboard com Abas ---
tab1, tab2, tab3 = st.tabs([
    "📍 Mapa de Risco", 
    "💨 Qualidade do Ar (Anomalias)", 
    "📊 Análise de Tendências (NLP)"
])

# --- Aba 1: Mapa de Risco ---
with tab1:
    st.header("Mapa de Indústrias por Risco Ambiental")
    st.write("Mapa de calor das autuações, classificado por risco (Alto, Médio, Baixo) usando o modelo de Random Forest.")
    
    df_mapa = df_autuacoes.dropna(subset=['NUM_LATITUDE_AUTO', 'NUM_LONGITUDE_AUTO', 'cor'])
    df_mapa = df_mapa.rename(columns={'NUM_LATITUDE_AUTO': 'lat', 'NUM_LONGITUDE_AUTO': 'lon'})
    
    if not df_mapa.empty:
        st.map(df_mapa,
               latitude='lat',
               longitude='lon',
               color='cor',
               zoom=3)
        st.info("Legenda de Risco: Vermelho (Alto), Laranja (Médio), Verde (Baixo)")
    else:
        st.warning("Nenhum dado de autuação com coordenadas válidas para exibir no mapa.")

# --- Aba 2: Qualidade do Ar e Alertas ---
with tab2:
    st.header("Índices de Qualidade do Ar e Alertas de Anomalia")
    
    if df_poluicao is not None and model_anomalia is not None:
        df_realtime = df_poluicao.iloc[-24:].copy()
        
        if not df_realtime.empty:
            df_realtime['anomalia'] = model_anomalia.predict(df_realtime)
            
            st.subheader("🚨 Alertas de Não Conformidade (Isolation Forest)")
            anomalias_detectadas = df_realtime[df_realtime['anomalia'] == -1]
            
            if anomalias_detectadas.empty:
                st.success("Nenhuma anomalia de poluição detectada nas últimas 24 horas.")
            else:
                st.warning(f"Alerta! {len(anomalias_detectadas)} anomalias detectadas nas últimas 24 horas.")
                st.dataframe(anomalias_detectadas)
                
            st.subheader("📈 Índices de Qualidade do Ar (Últimos Dados)")
            col1, col2, col3 = st.columns(3)
            
            last_record = df_poluicao.iloc[-1]
            
            col1.metric("CO (Monóxido de Carbono)", f"{last_record.get('CO(GT)', 0):.2f}", "mg/m³")
            col2.metric("NOx (Óxidos de Nitrogênio)", f"{last_record.get('NOx(GT)', 0):.2f}", "ppb")
            col3.metric("Temp. do Ar", f"{last_record.get('T', 0):.1f}", "°C")

            st.subheader("Histórico Recente (Últimas 24h)")
            cols_para_plotar = [col for col in ['CO(GT)', 'NOx(GT)', 'T'] if col in df_realtime.columns]
            if cols_para_plotar:
                st.line_chart(df_realtime[cols_para_plotar])
            else:
                st.warning("Colunas de poluição não encontradas para plotar o gráfico de linha.")
        else:
            st.warning("Não há dados de poluição recentes para exibir.")
    else:
        st.error("Dados de poluição ou modelo de anomalia não carregados.")

# --- Aba 3: Análise de Tendências (NLP) ---
with tab3:
    st.header("Análise de Tendências e Classificação (NLP)")
    
    if model_tema is not None:
        df_autuacoes['tematica'] = model_tema.predict(df_autuacoes['DES_INFRACAO'].fillna(""))
        
        st.subheader("Classificação da Temática das Autuações")
        
        fig1 = px.pie(df_autuacoes, 
                      names='tematica', 
                      title='Distribuição das Infrações por Tema (NLP)',
                      hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Análise de Tendências Temporais")
        
        df_trend = df_autuacoes.groupby(['ano', 'tematica']).size().reset_index(name='contagem')
        
        fig2 = px.bar(df_trend, 
                      x='ano', 
                      y='contagem', 
                      color='tematica',
                      title='Contagem de Autuações por Ano e Tema')
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.error("Modelo de temática (NLP) não carregado.")