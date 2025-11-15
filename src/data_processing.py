import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point # Para criar os pontos geográficos
import glob # Para encontrar os arquivos

# --- 1. Definição de Caminhos ---
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

IBAMA_DIR = os.path.join(RAW_PATH, "ibama_autuacoes_zip") 
UC_SHP_FILE = os.path.join(RAW_PATH, "unidades_conservacao_shp", "shp_cnuc_2024_10_pol.shp") 
POLLUTION_FILE = os.path.join(RAW_PATH, "poluicao_uci_historico.csv")

# CRS (Sistemas de Coordenadas)
CRS_GEOGRAFICO = "EPSG:4326" 
CRS_PROJETADO_METROS = "EPSG:5880" 

def load_ibama_data(directory_path):
    """
    Carrega TODOS os CSVs de autuação da pasta, 
    concatena, limpa e converte para GeoDataFrame.
    """
    print(f"Carregando dados do IBAMA de {directory_path}...")
    
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"Erro: Nenhum arquivo .csv encontrado em {directory_path}")
        return gpd.GeoDataFrame()

    all_dataframes = []
    
    for file_path in csv_files:
        try:
            df_temp = pd.read_csv(file_path, encoding='latin1', sep=';', low_memory=False)
        except UnicodeDecodeError:
            df_temp = pd.read_csv(file_path, encoding='utf-8', sep=';', low_memory=False)
        except Exception as e:
            print(f"  Erro ao ler {file_path}: {e}. Pulando este arquivo.")
            continue
            
        all_dataframes.append(df_temp)

    if not all_dataframes:
        print("Nenhum DataFrame do IBAMA foi carregado com sucesso.")
        return gpd.GeoDataFrame()

    df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Arquivos CSV do IBAMA concatenados. Total de registros brutos: {len(df)}")

    colunas_interesse = [
        'DAT_HORA_AUTO_INFRACAO', 'DES_INFRACAO', 'TP_PESSOA_INFRATOR',     
        'VAL_AUTO_INFRACAO', 'NUM_LATITUDE_AUTO', 'NUM_LONGITUDE_AUTO'      
    ]
    colunas_validas = [col for col in colunas_interesse if col in df.columns]
    df = df[colunas_validas]

    df['NUM_LATITUDE_AUTO'] = pd.to_numeric(
        df['NUM_LATITUDE_AUTO'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    df['NUM_LONGITUDE_AUTO'] = pd.to_numeric(
        df['NUM_LONGITUDE_AUTO'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )

    df = df.dropna(subset=['NUM_LATITUDE_AUTO', 'NUM_LONGITUDE_AUTO'])
    df = df[df['NUM_LATITUDE_AUTO'] != 0]

    print(f"Total de autuações com coordenadas válidas: {len(df)}")

    geometry = [Point(xy) for xy in zip(df['NUM_LONGITUDE_AUTO'], df['NUM_LATITUDE_AUTO'])]
    gdf_ibama = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_GEOGRAFICO)
    
    return gdf_ibama

def load_uc_data(file_path):
    """Carrega o Shapefile das Unidades de Conservação."""
    print(f"Carregando dados das UCs de {file_path}...")
    gdf_ucs = gpd.read_file(file_path, encoding='latin1')
    gdf_ucs = gdf_ucs.to_crs(CRS_GEOGRAFICO)
    print("Dados das UCs carregados.")
    return gdf_ucs

def calculate_proximity(gdf_ibama, gdf_ucs):
    """
    Calcula a distância de cada autuação à Unidade de Conservação (UC) mais próxima.
    """
    print("Iniciando cálculo de proximidade (pode levar alguns minutos)...")
    
    gdf_ibama_proj = gdf_ibama.to_crs(CRS_PROJETADO_METROS)
    gdf_ucs_proj = gdf_ucs.to_crs(CRS_PROJETADO_METROS)

    gdf_joined = gpd.sjoin_nearest(gdf_ibama_proj, gdf_ucs_proj, how='left')
    
    gdf_joined = gdf_joined[~gdf_joined.index.duplicated(keep='first')]

    gdf_joined_valid = gdf_joined.dropna(subset=['index_right'])
    
    if gdf_joined_valid.empty:
        print("Aviso: Nenhuma autuação foi mapeada para uma UC próxima.")
        gdf_ibama['distancia_uc_m'] = pd.NA
        gdf_ibama['nome_uc_proxima'] = pd.NA
        return gdf_ibama

    valid_uc_indices = gdf_joined_valid['index_right'].astype(int)
    uc_geometries = gdf_ucs_proj.loc[valid_uc_indices].geometry
    autuacoes_geometries = gdf_joined_valid.geometry
    
    distancias_calculadas = autuacoes_geometries.reset_index(drop=True).distance(
        uc_geometries.reset_index(drop=True)
    )

    distancias_series = pd.Series(distancias_calculadas.values, index=gdf_joined_valid.index)
    gdf_ibama['distancia_uc_m'] = distancias_series 

    nomes_series = gdf_ucs.loc[valid_uc_indices, 'nome_uc']
    nomes_series.index = gdf_joined_valid.index 
    gdf_ibama['nome_uc_proxima'] = nomes_series 

    print(f"Cálculo de proximidade concluído. {len(gdf_joined_valid)} autuações mapeadas.")
    return gdf_ibama

def process_features(gdf_ibama):
    """Cria features de Sazonalidade e Tipo de Indústria."""
    print("Criando features de Sazonalidade e Tipo de Indústria...")
    
    # Adiciona .copy() para evitar SettingWithCopyWarning
    gdf_ibama = gdf_ibama.copy()
    
    gdf_ibama['DAT_HORA_AUTO_INFRACAO'] = pd.to_datetime(gdf_ibama['DAT_HORA_AUTO_INFRACAO'], dayfirst=True, errors='coerce')
    gdf_ibama = gdf_ibama.dropna(subset=['DAT_HORA_AUTO_INFRACAO'])
    
    gdf_ibama['mes'] = gdf_ibama['DAT_HORA_AUTO_INFRACAO'].dt.month
    gdf_ibama['ano'] = gdf_ibama['DAT_HORA_AUTO_INFRACAO'].dt.year
    gdf_ibama['trimestre'] = gdf_ibama['DAT_HORA_AUTO_INFRACAO'].dt.quarter

    gdf_ibama['tipo_industria'] = gdf_ibama['TP_PESSOA_INFRATOR'].astype(str).apply(
        lambda x: 'Juridica' if 'Jurídica' in str(x) else 'Fisica'
    )
    
    if 'VAL_AUTO_INFRACAO' in gdf_ibama.columns:
        gdf_ibama['valor_multa'] = pd.to_numeric(
            gdf_ibama['VAL_AUTO_INFRACAO'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), 
            errors='coerce'
        )
        gdf_ibama = gdf_ibama.drop(columns=['VAL_AUTO_INFRACAO'])

    return gdf_ibama

def process_pollution_data(file_path):
    """Limpa o dataset de poluição (UCI) para o Isolation Forest."""
    print(f"Processando dados de poluição de {file_path}...")
    try:
        df_pol = pd.read_csv(file_path, sep=';', decimal=',')
    except Exception as e:
        print(f"Erro ao ler arquivo de poluição: {e}")
        return

    df_pol = df_pol.replace(-200, pd.NA)
    
    if 'Date' not in df_pol.columns or 'Time' not in df_pol.columns:
        print("Erro: Colunas 'Date' ou 'Time' não encontradas no arquivo de poluição.")
        if pd.api.types.is_datetime64_any_dtype(df_pol.iloc[:, 0]):
             df_pol = df_pol.set_index(df_pol.columns[0])
        else:
             return 
    else:
        # <--- CORREÇÃO (v10) ---
        # 1. Cria a coluna datetime, 'coerce' transforma falhas em NaT
        df_pol['datetime'] = pd.to_datetime(df_pol['Date'] + ' ' + df_pol['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
        
        # 2. Remove as linhas onde a conversão para NaT falhou
        df_pol = df_pol.dropna(subset=['datetime'])
        
        # 3. Define o índice limpo
        df_pol = df_pol.set_index('datetime')
        
        # 4. Ordena o índice (essencial para interpolação 'time')
        df_pol = df_pol.sort_index()
        # <--- FIM DA CORREÇÃO ---
    
    colunas_sensores = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
        'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH'
    ]
    colunas_validas = [col for col in colunas_sensores if col in df_pol.columns]
    df_pol_clean = df_pol[colunas_validas].copy()
    
    print("Convertendo colunas de poluição para numérico...")
    for col in df_pol_clean.columns:
        df_pol_clean[col] = pd.to_numeric(df_pol_clean[col], errors='coerce')
    
    print("Interpolando dados de poluição...")
    # Agora a interpolação deve funcionar
    df_pol_clean = df_pol_clean.interpolate(method='time')
    df_pol_clean = df_pol_clean.bfill()
    
    save_path = os.path.join(PROCESSED_PATH, "poluicao_processada.parquet")
    df_pol_clean.to_parquet(save_path)
    print(f"Dados de poluição salvos em {save_path}")

def main():
    """Orquestrador do pipeline de processamento de dados."""
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    print("--- Iniciando Pipeline de Processamento de Dados (v10) ---")
    
    # --- Parte 1: Autuações (Já deve estar concluída) ---
    autuacoes_save_path = os.path.join(PROCESSED_PATH, "autuacoes_processadas.parquet")
    if not os.path.exists(autuacoes_save_path):
        print("Processando dados de autuações...")
        gdf_ibama = load_ibama_data(IBAMA_DIR)
        
        if gdf_ibama.empty:
            print("Nenhum dado do IBAMA foi carregado. Abortando o processamento.")
            return 
            
        gdf_ucs = load_uc_data(UC_SHP_FILE)
        
        gdf_ibama = calculate_proximity(gdf_ibama, gdf_ucs)
        
        gdf_ibama = process_features(gdf_ibama)
        
        df_final_autuacoes = pd.DataFrame(gdf_ibama.drop(columns='geometry'))
        df_final_autuacoes.to_parquet(autuacoes_save_path)
        print(f"Dados de autuações processados e salvos em {autuacoes_save_path}")
    else:
        print("Arquivo 'autuacoes_processadas.parquet' já existe. Pulando etapa 1.")

    # --- Parte 2: Poluição (Onde o script falhou) ---
    process_pollution_data(POLLUTION_FILE)
    
    print("--- Pipeline de Processamento Concluído ---")

if __name__ == "__main__":
    main()