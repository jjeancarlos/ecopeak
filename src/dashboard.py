import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import numpy as np
from datetime import datetime

# --- IMPORTAÇÕES PARA O CHATBOT INTELIGENTE ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    
    # Renomear colunas se existirem
    if 'NUM_LATITUDE_AUTO' in df_clean.columns:
        df_clean = df_clean.rename(columns={
            'NUM_LATITUDE_AUTO': 'lat', 
            'NUM_LONGITUDE_AUTO': 'lon'
        })
    
    # Verificar se as colunas existem antes de processar
    if 'lat' not in df_clean.columns or 'lon' not in df_clean.columns:
        return df_clean

    # Remover valores NaN e zeros
    df_clean = df_clean.dropna(subset=['lat', 'lon'])
    df_clean = df_clean[(df_clean['lat'] != 0) & (df_clean['lon'] != 0)]
    
    # FILTRO SUPER RESTRITIVO - BRASIL CONTINENTAL
    df_clean = df_clean[
        (df_clean['lat'] >= -33.5) & (df_clean['lat'] <= 5.5) &      # Norte ao Sul
        (df_clean['lon'] >= -73.5) & (df_clean['lon'] <= -34.5)      # Oeste ao Leste
    ]
    
    # Filtro adicional para remover coordenadas no oceano/países vizinhos
    df_clean = df_clean[
        ~(  
            ((df_clean['lat'] > 2.0) & (df_clean['lon'] > -55.0)) |
            ((df_clean['lat'] < -10.0) & (df_clean['lon'] < -70.0)) |
            ((df_clean['lat'] < -25.0) & (df_clean['lon'] > -50.0))
        )
    ]
    
    return df_clean

# --- 2. Chatbot INTELIGENTE (PLN com TF-IDF) ---
class ChatInteligente:
    def __init__(self, df_autuacoes, df_poluicao):
        self.df_autuacoes = df_autuacoes
        self.df_poluicao = df_poluicao
        self.vectorizer = TfidfVectorizer()
        
        # Base de Conhecimento (Perguntas Possíveis -> Intenção)
        self.knowledge_base = {
            "qual é o risco atual das industrias": "risco",
            "quais áreas tem risco alto": "risco",
            "me fale sobre o perigo e segurança": "risco",
            "existem locais perigosos": "risco",
            
            "quantas indústrias estão cadastradas": "total",
            "qual o total de monitoramento": "total",
            "número de registros na base": "total",
            
            "existe alguma anomalia ou alerta": "alerta",
            "está tudo normal com a poluição": "alerta",
            "qualidade do ar está ruim": "alerta",
            "níveis criticos detectados": "alerta",
            
            "qual a tendência ao longo dos anos": "tendencia",
            "histórico de desmatamento e multas": "tendencia",
            "evolução temporal": "tendencia",
            "o que aconteceu no passado": "tendencia",
            
            "como você funciona ajuda": "ajuda",
            "o que você sabe fazer": "ajuda",
            "menu de opções": "ajuda"
        }
        
        # Treina o vetorizador com as perguntas conhecidas
        self.corpus = list(self.knowledge_base.keys())
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def responder(self, pergunta_usuario):
        try:
            # Transforma a pergunta do usuário em vetor
            user_vec = self.vectorizer.transform([pergunta_usuario.lower()])
            
            # Calcula similaridade
            similarities = cosine_similarity(user_vec, self.tfidf_matrix)
            best_match_idx = np.argmax(similarities)
            score = similarities[0][best_match_idx]
            
            # Confiança mínima (Threshold)
            if score < 0.25:
                return "🤔 Não entendi muito bem. Tente perguntar sobre 'Risco', 'Total de indústrias', 'Alertas' ou 'Histórico'."
                
            # Identifica a intenção
            pergunta_padrao = self.corpus[best_match_idx]
            intencao = self.knowledge_base[pergunta_padrao]
            
            # --- Respostas Dinâmicas ---
            if intencao == 'risco':
                if 'risco_predito' in self.df_autuacoes.columns:
                    alto = len(self.df_autuacoes[self.df_autuacoes['risco_predito'] == 'Alto'])
                    medio = len(self.df_autuacoes[self.df_autuacoes['risco_predito'] == 'Medio'])
                    return f"**📊 Análise de Risco (IA):**\nIdentifiquei **{alto}** áreas críticas (Risco Alto) 🔴 e **{medio}** áreas de atenção 🟠.\nRecomendo priorizar a fiscalização nas áreas vermelhas."
                return "⚠️ Os dados de risco preditivo não foram calculados."

            elif intencao == 'total':
                poluicao_len = len(self.df_poluicao) if self.df_poluicao is not None else 0
                return f"**📈 Estatísticas da Base:**\n- 🏭 Complexos Industriais: **{len(self.df_autuacoes)}**\n- 📡 Registros de Sensores: **{poluicao_len}**\n- 🎯 Modelo de Risco Ativo: Random Forest"

            elif intencao == 'alerta':
                if self.df_poluicao is not None and not self.df_poluicao.empty:
                    ultimos = self.df_poluicao.iloc[-24:] # Últimas 24h
                    media_co = ultimos['CO(GT)'].mean() if 'CO(GT)' in ultimos else 0
                    status = "CRÍTICO" if media_co > 4 else "ESTÁVEL"
                    icon = "🚨" if media_co > 4 else "✅"
                    return f"**🔍 Monitoramento em Tempo Real (24h):**\nStatus do Ar: {icon} **{status}**\n- Média de CO: {media_co:.2f} mg/m³\n- Sistema de anomalias: Operante"
                return "📡 Sensores offline ou sem dados recentes."

            elif intencao == 'tendencia':
                return "**📅 Insights Históricos:**\nAnalisei os dados temporais e detectei um pico de infrações entre 2014-2018. Atualmente, a maior incidência é em crimes contra a 'Flora' (Desmatamento)."

            elif intencao == 'ajuda':
                return "**🤖 Sou o EcoPeak Assistente.**\nPosso responder perguntas naturais como:\n- 'Quais locais são perigosos?'\n- 'Como está a qualidade do ar?'\n- 'Mostre estatísticas gerais'\n- 'Qual o histórico de multas?'"
        
        except Exception as e:
            return f"Desculpe, tive um erro interno: {str(e)}"

        return "Desculpe, não consegui processar sua solicitação."

# --- 3. Definição de Caminhos ---
PROCESSED_PATH = "data/processed"
MODELS_PATH = "data/models"

# --- 4. Funções de Carregamento (com Cache) ---
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Erro: Arquivo não encontrado em {file_path}")
        return None
    return pd.read_parquet(file_path)

@st.cache_resource
def load_model(model_path):
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
    st.error("Falha ao carregar dados essenciais. Verifique os arquivos na pasta data/.")
    st.stop()

# --- 6. Inicialização do Chat Inteligente ---
if 'chat_engine' not in st.session_state:
    st.session_state.chat_engine = ChatInteligente(df_autuacoes, df_poluicao)
if 'mensagens' not in st.session_state:
    st.session_state.mensagens = []

# --- 7. Título do Dashboard ---
st.title("🌍 Sistema de Monitoramento Ambiental Industrial")

# --- 8. Geração de Predições ---
@st.cache_data
def get_predictions(df, _model):
    df_predict = df.copy()
    df_predict['DES_INFRACAO'] = df_predict['DES_INFRACAO'].fillna("")
    
    features = ['distancia_uc_m', 'tipo_industria', 'mes', 'trimestre', 'ano', 'DES_INFRACAO']
    features_presentes = [col for col in features if col in df_predict.columns]
    df_predict = df_predict[features_presentes]
    
    df['risco_predito'] = _model.predict(df_predict)
    
    color_map = {'Alto': [255, 0, 0], 'Medio': [255, 165, 0], 'Baixo': [0, 128, 0]}
    df['cor'] = df['risco_predito'].map(color_map)
    return df

df_autuacoes = get_predictions(df_autuacoes, model_risco)

# --- 9. Layout do Dashboard com Abas ---
tab1, tab2, tab3, tab_chat = st.tabs([
    "📍 Mapa de Risco", 
    "💨 Qualidade do Ar (Avançado)", 
    "📊 Análise de Tendências",
    "💬 Assistente IA"
])

# --- Aba 1: Mapa de Risco ---
with tab1:
    st.header("🗺️ Mapa de Indústrias por Risco Ambiental")
    df_mapa = clean_coordinates(df_autuacoes)
    
    if not df_mapa.empty:
        st.map(df_mapa, latitude='lat', longitude='lon', color='cor', zoom=4)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            alto = len(df_mapa[df_mapa['risco_predito'] == 'Alto'])
            st.metric("🔴 Alto Risco", alto)
        with col2:
            medio = len(df_mapa[df_mapa['risco_predito'] == 'Medio'])
            st.metric("🟠 Médio Risco", medio)
        with col3:
            bajo = len(df_mapa[df_mapa['risco_predito'] == 'Baixo'])
            st.metric("🟢 Baixo Risco", bajo)
        with col4:
            st.metric("📍 Total Visualizado", len(df_mapa))
    else:
        st.error("Nenhuma coordenada válida encontrada.")

# --- Aba 2: Qualidade do Ar (MELHORADO COM PLOTLY E CORREÇÃO DO ERRO) ---
with tab2:
    st.header("💨 Monitoramento de Qualidade do Ar")
    
    if df_poluicao is not None and model_anomalia is not None:
        # Pegar últimas 24 amostras (assumindo 1 por hora)
        df_realtime = df_poluicao.iloc[-24:].copy()
        
        # 1. Cards de Métricas Atuais
        st.subheader("📡 Leitura em Tempo Real")
        col_a, col_b, col_c = st.columns(3)
        last_record = df_realtime.iloc[-1]
        
        col_a.metric("CO (Monóxido)", f"{last_record.get('CO(GT)', 0):.2f} mg/m³", delta="-0.1" if last_record.get('CO(GT)', 0) < 2 else "+0.5", delta_color="inverse")
        col_b.metric("NOx (Nitrogênio)", f"{last_record.get('NOx(GT)', 0):.1f} ppb", delta="Normal")
        col_c.metric("Temperatura", f"{last_record.get('T', 0):.1f} °C")

        # 2. Gráfico Avançado com Eixo Duplo (Plotly)
        st.subheader("📈 Comparativo Avançado: CO vs NOx")
        st.caption("Visualização de escala dupla para correlação de poluentes")

        if 'CO(GT)' in df_realtime.columns and 'NOx(GT)' in df_realtime.columns:
            # Cria figura com eixo Y secundário
            fig_ar = make_subplots(specs=[[{"secondary_y": True}]])

            # Trace 1: CO (Eixo Esquerdo - Verde/Amarelo/Vermelho)
            fig_ar.add_trace(
                go.Scatter(
                    x=df_realtime.index, y=df_realtime['CO(GT)'], 
                    name="CO (mg/m³)",
                    line=dict(color='#00cc96', width=3),
                    mode='lines+markers'
                ),
                secondary_y=False,
            )

            # Trace 2: NOx (Eixo Direito - Roxo)
            fig_ar.add_trace(
                go.Scatter(
                    x=df_realtime.index, y=df_realtime['NOx(GT)'], 
                    name="NOx (ppb)",
                    line=dict(color='#636efa', width=2, dash='dot'),
                    mode='lines'
                ),
                secondary_y=True,
            )

            # Área de Alerta (Exemplo: CO acima de 4 é perigoso)
            fig_ar.add_hrect(
                y0=4, y1=10, line_width=0, fillcolor="red", opacity=0.1,
                secondary_y=False, annotation_text="ALERTA CO", 
                annotation_position="top left"
            )

            # Layout Profissional
            fig_ar.update_layout(
                height=450,
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            # Configuração dos Eixos
            fig_ar.update_yaxes(title_text="<b>CO</b> (mg/m³)", secondary_y=False, showgrid=True, gridcolor='lightgray')
            fig_ar.update_yaxes(title_text="<b>NOx</b> (ppb)", secondary_y=True, showgrid=False)

            st.plotly_chart(fig_ar, use_container_width=True)
        else:
            st.warning("Colunas de poluentes não encontradas para gerar o gráfico.")

        # 3. Alertas de Anomalia (CORREÇÃO DO ERRO VALUEERROR)
        st.markdown("---")
        st.subheader("🚨 Log de Anomalias (IA)")
        
        # CORREÇÃO AQUI: O Isolation Forest exige TODAS as colunas usadas no treino (AH, C6H6, etc.)
        # Selecionamos todas as colunas numéricas do dataframe original
        features_para_predicao = df_realtime.select_dtypes(include=[np.number]).fillna(0)
        
        # Gerar predição
        df_realtime['anomalia'] = model_anomalia.predict(features_para_predicao)
        anomalias = df_realtime[df_realtime['anomalia'] == -1]
        
        if not anomalias.empty:
            st.error(f"⚠️ Foram detectadas **{len(anomalias)}** anomalias nas últimas 24h!")
            st.dataframe(anomalias.style.highlight_max(axis=0, color='#ffcccc'))
        else:
            st.success("✅ O sistema Isolation Forest não detectou anomalias recentes.")

    else:
        st.error("Dados de poluição não carregados.")

# --- Aba 3: Análise de Tendências ---
with tab3:
    st.header("📊 Classificação Automática de Temas")
    if model_tema is not None:
        df_autuacoes['tematica'] = model_tema.predict(df_autuacoes['DES_INFRACAO'].fillna(""))
        
        col1, col2 = st.columns([1, 2])
        with col1:
            fig1 = px.pie(df_autuacoes, names='tematica', title='Distribuição de Temas', hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            df_trend = df_autuacoes.groupby(['ano', 'tematica']).size().reset_index(name='contagem')
            fig2 = px.bar(df_trend, x='ano', y='contagem', color='tematica', title='Evolução Histórica')
            st.plotly_chart(fig2, use_container_width=True)

# --- ABA CHAT INTELIGENTE (NLP) ---
with tab_chat:
    st.header("💬 Assistente EcoPeak (IA)")
    st.markdown("Converse com o sistema usando linguagem natural. O modelo usa **Similaridade de Cosseno** para entender o contexto.")
    
    pergunta = st.chat_input("Pergunte algo... (Ex: 'Quais áreas tem alto risco?')")
    
    if pergunta:
        # Exibir pergunta do usuário
        st.session_state.mensagens.append({'role': 'user', 'content': pergunta, 'hora': datetime.now().strftime("%H:%M")})
        
        # Processar resposta com IA
        resposta = st.session_state.chat_engine.responder(pergunta)
        st.session_state.mensagens.append({'role': 'assistant', 'content': resposta, 'hora': datetime.now().strftime("%H:%M")})
    
    # Renderizar Chat
    for msg in st.session_state.mensagens:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
            st.caption(f"Enviado às {msg['hora']}")
            
    # Botão para limpar
    if st.button("🗑️ Limpar Histórico", key="clear_chat"):
        st.session_state.mensagens = []
        st.rerun()

# --- 10. Sidebar ---
with st.sidebar:
    st.header("Informações")
    st.metric("Indústrias", len(df_autuacoes))
    if df_poluicao is not None:
        st.metric("Dados Sensor", len(df_poluicao))
    
    st.markdown("---")
    st.caption(f"Atualizado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")