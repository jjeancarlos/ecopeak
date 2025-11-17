import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import json
from datetime import datetime

# --- 1. Configuração da Página ---
st.set_page_config(
    page_title="Monitoramento Ambiental",
    page_icon="🌍",
    layout="wide"
)

# --- FUNÇÃO OTIMIZADA PARA LIMPEZA DE COORDENADAS ---
def clean_coordinates(df):
    """Filtro agressivo para manter APENAS coordenadas válidas do Brasil continental"""
    df_clean = df.copy()
    
    # Renomear colunas
    df_clean = df_clean.rename(columns={
        'NUM_LATITUDE_AUTO': 'lat', 
        'NUM_LONGITUDE_AUTO': 'lon'
    })
    
    # Remover valores NaN e zeros
    df_clean = df_clean.dropna(subset=['lat', 'lon'])
    df_clean = df_clean[(df_clean['lat'] != 0) & (df_clean['lon'] != 0)]
    
    # FILTRO SUPER RESTRITIVO - BRASIL CONTINENTAL
    # Coordenadas aproximadas do território brasileiro
    df_clean = df_clean[
        (df_clean['lat'] >= -33.5) & (df_clean['lat'] <= 5.5) &      # Norte ao Sul
        (df_clean['lon'] >= -73.5) & (df_clean['lon'] <= -34.5)      # Oeste ao Leste
    ]
    
    # Filtro adicional para remover coordenadas no oceano/países vizinhos
    df_clean = df_clean[
        ~(  # REMOVER estas áreas problemáticas:
            # Área do Caribe/Norte da América do Sul
            ((df_clean['lat'] > 2.0) & (df_clean['lon'] > -55.0)) |
            # Área do Pacífico/Oeste
            ((df_clean['lat'] < -10.0) & (df_clean['lon'] < -70.0)) |
            # Área da Argentina/Extremo Sul
            ((df_clean['lat'] < -25.0) & (df_clean['lon'] > -50.0))
        )
    ]
    
    return df_clean

# --- 2. Chatbot SIMPLES ---
class ChatSimples:
    def __init__(self, df_autuacoes, df_poluicao):
        self.df_autuacoes = df_autuacoes
        self.df_poluicao = df_poluicao
    
    def responder(self, pergunta):
        pergunta = pergunta.lower()
        
        if 'risco' in pergunta:
            if 'risco_predito' in self.df_autuacoes.columns:
                alto = len(self.df_autuacoes[self.df_autuacoes['risco_predito'] == 'Alto'])
                medio = len(self.df_autuacoes[self.df_autuacoes['risco_predito'] == 'Medio'])
                baixo = len(self.df_autuacoes[self.df_autuacoes['risco_predito'] == 'Baixo'])
                return f"**📊 Risco Atual:**\n- 🔴 Alto: {alto} áreas\n- 🟠 Médio: {medio} áreas\n- 🟢 Baixo: {baixo} áreas\n- 📍 Total: {len(self.df_autuacoes)} indústrias"
            return "⚠️ Dados de risco não disponíveis"
        
        elif 'quantas' in pergunta or 'total' in pergunta:
            poluicao_count = len(self.df_poluicao) if self.df_poluicao is not None else 0
            return f"**📈 Estatísticas:**\n- 🏭 Indústrias: {len(self.df_autuacoes)}\n- 🌫️ Registros poluição: {poluicao_count}\n- 🎯 Precisão: 61%"
        
        elif 'anomalia' in pergunta or 'alerta' in pergunta:
            if self.df_poluicao is not None:
                return f"**🔍 Sistema de Anomalias:**\n- ✅ Monitorando {len(self.df_poluicao)} registros\n- 📡 Sistema operacional\n- ⚠️ Detecção em tempo real"
            return "🌫️ Dados de poluição não carregados"
        
        elif 'tendência' in pergunta or 'histórico' in pergunta:
            if 'ano' in self.df_autuacoes.columns:
                anos = self.df_autuacoes['ano'].nunique()
                return f"**📅 Análise Temporal:**\n- 📊 {anos} anos de dados\n- 📈 Pico 2014-2018\n- 🌿 Flora/Desmatamento predominante"
            return "📊 Dados históricos não disponíveis"
        
        elif 'ajuda' in pergunta:
            return "**🤖 Como usar:**\nPergunte sobre:\n- ❓ 'risco atual'\n- ❓ 'quantas indústrias'  \n- ❓ 'alertas/anomalias'\n- ❓ 'tendência histórica'\n- ❓ 'estatísticas'"
        
        else:
            return "🤖 EcoPeak: Pergunte sobre 'risco', 'quantas indústrias', 'alertas' ou digite 'ajuda'"

# --- 3. Definição de Caminhos ---
PROCESSED_PATH = "data/processed"
MODELS_PATH = "data/models"

# --- 4. Funções de Carregamento (com Cache) ---
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

# --- 5. Carregamento dos Dados e Modelos ---
with st.spinner('Carregando dados e modelos...'):
    df_autuacoes = load_data(os.path.join(PROCESSED_PATH, "autuacoes_processadas.parquet"))
    df_poluicao = load_data(os.path.join(PROCESSED_PATH, "poluicao_processada.parquet"))
    
    model_risco = load_model(os.path.join(MODELS_PATH, "random_forest_risk_pipeline.joblib"))
    model_anomalia = load_model(os.path.join(MODELS_PATH, "isolation_forest.joblib"))
    model_tema = load_model(os.path.join(MODELS_PATH, "nlp_topic_pipeline.joblib"))

if df_autuacoes is None or model_risco is None:
    st.error("Falha ao carregar dados essenciais ou modelo de risco. Verifique os caminhos.")
    st.stop()

# --- 6. Inicialização do Chat SIMPLES ---
if 'chat_simples' not in st.session_state:
    st.session_state.chat_simples = ChatSimples(df_autuacoes, df_poluicao)
if 'mensagens' not in st.session_state:
    st.session_state.mensagens = []

# --- 7. Título do Dashboard ---
st.title("🌍 Sistema de Monitoramento Ambiental Industrial")

# --- 8. Geração de Predições ---
@st.cache_data
def get_predictions(df, _model):
    """Gera predições de risco para o dataframe de autuações."""
    df_predict = df.copy()
    df_predict['DES_INFRACAO'] = df_predict['DES_INFRACAO'].fillna("")
    
    features = [
        'distancia_uc_m', 'tipo_industria', 'mes',            
        'trimestre', 'ano', 'DES_INFRACAO'
    ]
    
    features_presentes = [col for col in features if col in df_predict.columns]
    df_predict = df_predict[features_presentes]
    
    df['risco_predito'] = _model.predict(df_predict)
    
    color_map = {
        'Alto': [255, 0, 0],
        'Medio': [255, 165, 0],
        'Baixo': [0, 128, 0]
    }
    df['cor'] = df['risco_predito'].map(color_map)
    return df

# Gera predições de risco para o mapa
df_autuacoes = get_predictions(df_autuacoes, model_risco)

# --- 9. Layout do Dashboard com Abas ---
tab1, tab2, tab3, tab_chat = st.tabs([
    "📍 Mapa de Risco", 
    "💨 Qualidade do Ar", 
    "📊 Análise de Tendências",
    "💬 Chat Simples"
])

# --- Aba 1: Mapa de Risco (OTIMIZADO) ---
with tab1:
    st.header("🗺️ Mapa de Indústrias por Risco Ambiental")
    st.write("Mapa de calor das autuações classificadas por risco (Alto, Médio, Baixo)")
    
    # Aplicar filtro agressivo
    df_mapa = clean_coordinates(df_autuacoes)
    
    if not df_mapa.empty:
        st.success(f"✅ **Mostrando {len(df_mapa)} localizações válidas no território brasileiro**")
        
        # Mapa com zoom otimizado para Brasil
        st.map(df_mapa,
               latitude='lat',
               longitude='lon',
               color='cor',
               zoom=4)
        
        # Estatísticas em tempo real
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            alto = len(df_mapa[df_mapa['risco_predito'] == 'Alto'])
            st.metric("🔴 Alto Risco", alto, delta=f"{alto} áreas")
        with col2:
            medio = len(df_mapa[df_mapa['risco_predito'] == 'Medio'])
            st.metric("🟠 Médio Risco", medio, delta=f"{medio} áreas")
        with col3:
            baixo = len(df_mapa[df_mapa['risco_predito'] == 'Baixo'])
            st.metric("🟢 Baixo Risco", baixo, delta=f"{baixo} áreas")
        with col4:
            st.metric("📍 Total Mapeado", len(df_mapa))
            
        st.info("**Legenda:** 🔴 Alto Risco | 🟠 Médio Risco | 🟢 Baixo Risco")
        
        # Informações de debug (opcional)
        with st.expander("🔍 Detalhes Técnicos"):
            st.write(f"**Coordenadas filtradas:** {len(df_mapa)} de {len(df_autuacoes)} total")
            st.write(f"**Extensão geográfica:**")
            st.write(f"- Latitude: {df_mapa['lat'].min():.2f}° a {df_mapa['lat'].max():.2f}°")
            st.write(f"- Longitude: {df_mapa['lon'].min():.2f}° a {df_mapa['lon'].max():.2f}°")
            
    else:
        st.error("❌ Nenhuma coordenada válida encontrada após filtro")
        
        # Diagnóstico detalhado
        with st.expander("🔧 Diagnóstico do Problema"):
            st.write("**Análise das coordenadas originais:**")
            if 'NUM_LATITUDE_AUTO' in df_autuacoes.columns:
                coords_originais = df_autuacoes[['NUM_LATITUDE_AUTO', 'NUM_LONGITUDE_AUTO']].dropna()
                st.write(f"- Coordenadas não-nulas: {len(coords_originais)}")
                st.write(f"- Latitude range: {coords_originais['NUM_LATITUDE_AUTO'].min():.2f} a {coords_originais['NUM_LATITUDE_AUTO'].max():.2f}")
                st.write(f"- Longitude range: {coords_originais['NUM_LONGITUDE_AUTO'].min():.2f} a {coords_originais['NUM_LONGITUDE_AUTO'].max():.2f}")
                
                # Mostrar amostra problemática
                st.write("**Amostra de coordenadas problemáticas:**")
                st.dataframe(coords_originais.head(10))

# --- Aba 2: Qualidade do Ar e Alertas ---
with tab2:
    st.header("💨 Qualidade do Ar e Alertas de Anomalia")
    
    if df_poluicao is not None and model_anomalia is not None:
        df_realtime = df_poluicao.iloc[-24:].copy()
        
        if not df_realtime.empty:
            df_realtime['anomalia'] = model_anomalia.predict(df_realtime)
            
            st.subheader("🚨 Alertas de Não Conformidade")
            anomalias_detectadas = df_realtime[df_realtime['anomalia'] == -1]
            
            if anomalias_detectadas.empty:
                st.success("✅ Nenhuma anomalia de poluição detectada nas últimas 24 horas.")
            else:
                st.error(f"⚠️ ALERTA! {len(anomalias_detectadas)} anomalias detectadas nas últimas 24 horas.")
                st.dataframe(anomalias_detectadas)
                
            st.subheader("📊 Índices de Qualidade do Ar")
            col1, col2, col3 = st.columns(3)
            
            last_record = df_poluicao.iloc[-1]
            
            col1.metric("CO (Monóxido)", f"{last_record.get('CO(GT)', 0):.2f}", "mg/m³")
            col2.metric("NOx (Nitrogênio)", f"{last_record.get('NOx(GT)', 0):.2f}", "ppb")
            col3.metric("Temperatura", f"{last_record.get('T', 0):.1f}", "°C")

            st.subheader("📈 Histórico Recente (24h)")
            cols_para_plotar = [col for col in ['CO(GT)', 'NOx(GT)', 'T'] if col in df_realtime.columns]
            if cols_para_plotar:
                st.line_chart(df_realtime[cols_para_plotar])
            else:
                st.warning("Colunas de poluição não encontradas para plotar o gráfico.")
        else:
            st.warning("Não há dados de poluição recentes para exibir.")
    else:
        st.error("Dados de poluição ou modelo de anomalia não carregados.")

# --- Aba 3: Análise de Tendências (NLP) ---
with tab3:
    st.header("📊 Análise de Tendências e Classificação")
    
    if model_tema is not None:
        df_autuacoes['tematica'] = model_tema.predict(df_autuacoes['DES_INFRACAO'].fillna(""))
        
        st.subheader("🎯 Distribuição por Temática")
        fig1 = px.pie(df_autuacoes, 
                      names='tematica', 
                      title='Distribuição das Infrações por Tema',
                      hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("📅 Evolução Temporal")
        df_trend = df_autuacoes.groupby(['ano', 'tematica']).size().reset_index(name='contagem')
        fig2 = px.bar(df_trend, 
                      x='ano', 
                      y='contagem', 
                      color='tematica',
                      title='Evolução das Autuações por Ano e Tema')
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.error("Modelo de classificação de temas não carregado.")

# --- ABA CHAT SIMPLES ---
with tab_chat:
    st.header("💬 Assistente EcoPeak")
    st.write("Faça perguntas em linguagem natural sobre os dados ambientais")
    
    # Input e botão
    pergunta = st.text_input(
        "**Digite sua pergunta:**",
        placeholder="Ex: Qual o risco atual? Quantas indústrias monitoradas?",
        key="input_chat"
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📤 Enviar Pergunta", use_container_width=True) and pergunta:
            resposta = st.session_state.chat_simples.responder(pergunta)
            st.session_state.mensagens.append({
                'pergunta': pergunta,
                'resposta': resposta,
                'hora': datetime.now().strftime("%H:%M")
            })
            st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.rerun()
    
    # Histórico do chat
    st.markdown("---")
    st.subheader("💭 Conversa")
    
    if not st.session_state.mensagens:
        st.info("""
        **💡 Exemplos de perguntas:**
        - "Qual o risco atual?"
        - "Quantas indústrias estão monitoradas?"  
        - "Há alertas de anomalia?"
        - "Qual a tendência histórica?"
        - "Mostre estatísticas"
        """)
    else:
        for msg in reversed(st.session_state.mensagens[-6:]):
            with st.chat_message("user"):
                st.write(f"**Você** ({msg['hora']}): {msg['pergunta']}")
            with st.chat_message("assistant"):
                st.write(f"**EcoPeak** ({msg['hora']}): {msg['resposta']}")
            st.markdown("---")

# --- 10. Sidebar com Informações ---
with st.sidebar:
    st.header("ℹ️ Informações do Sistema")
    
    # Métricas principais
    st.metric("🏭 Indústrias", len(df_autuacoes))
    if df_poluicao is not None:
        st.metric("🌫️ Dados Poluição", len(df_poluicao))
    
    # Estatísticas de risco
    st.markdown("---")
    st.subheader("🎯 Níveis de Risco")
    if 'risco_predito' in df_autuacoes.columns:
        risco_alto = len(df_autuacoes[df_autuacoes['risco_predito'] == 'Alto'])
        risco_medio = len(df_autuacoes[df_autuacoes['risco_predito'] == 'Medio'])
        risco_baixo = len(df_autuacoes[df_autuacoes['risco_predito'] == 'Baixo'])
        
        st.write(f"🔴 **Alto:** {risco_alto}")
        st.write(f"🟠 **Médio:** {risco_medio}") 
        st.write(f"🟢 **Baixo:** {risco_baixo}")
    
    # Informações técnicas
    st.markdown("---")
    st.markdown("**⚙️ Especificações:**")
    st.markdown("- Precisão: 61%")
    st.markdown("- Fator Principal: Distância até UCs")
    st.markdown("- Modelo: Random Forest")
    
    st.markdown("---")
    st.markdown("🔄 **Atualizado em:**")
    st.write(datetime.now().strftime("%d/%m/%Y %H:%M"))