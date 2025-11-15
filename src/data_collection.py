import os
import requests
import zipfile
import io # Usado para manipular o zip em memória

# Define o caminho base para salvar os dados
RAW_DATA_PATH = "data/raw"

def _download_file(url, filename, encoding=None):
    """Função auxiliar para baixar e salvar um arquivo CSV."""
    save_path = os.path.join(RAW_DATA_PATH, filename)
    try:
        print(f"Baixando {filename} de {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() 

        # Define a codificação se especificada
        if encoding:
            response.encoding = encoding
            
        with open(save_path, "w", encoding=encoding if encoding else 'utf-8') as f:
            f.write(response.text)
            
        print(f"Arquivo salvo em: {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar {url}: {e}")

def _download_zip(url, extract_folder_name):
    """Função auxiliar para baixar um arquivo ZIP e extraí-lo."""
    extract_path = os.path.join(RAW_DATA_PATH, extract_folder_name)
    try:
        print(f"Baixando e extraindo ZIP de {url}...")
        response = requests.get(url)
        response.raise_for_status()

        # Extrai o zip em memória
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extract_path)
            
        print(f"Arquivos extraídos para: {extract_path}")

    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar {url}: {e}")
    except zipfile.BadZipFile:
        print(f"Erro: O arquivo baixado de {url} não é um ZIP válido.")

def fetch_ibama_data():
    """
    Busca dados de autos de infração (multas) do IBAMA.
    Fonte: Novo link (dadosabertos.ibama.gov.br) - agora é um ZIP.
    """
    # NOVO LINK: Este é um ZIP que contém vários CSVs sobre autuações
    url = "https://dadosabertos.ibama.gov.br/dados/SIFISC/auto_infracao/auto_infracao/auto_infracao_csv.zip"
    _download_zip(url, "ibama_autuacoes_zip")

def fetch_unidades_conservacao():
    """
    Busca os Shapefiles (.shp) das Unidades de Conservação (UCs) do MMA/ICMBio.
    """
    # NOVO LINK: Portal de dados do MMA (Out/2024)
    url = "https://dados.mma.gov.br/dataset/44b6dc8a-dc82-4a84-8d95-1b0da7c85dac/resource/7a142cc0-dae9-4a0b-8180-3016994d2932/download/shp_cnuc_2024_10.zip"
    _download_zip(url, "unidades_conservacao_shp")

def fetch_poluicao_data():
    """
    Busca dados de poluição.
    O link da CETESB é instável. Usaremos um dataset clássico da UCI 
    (Air Quality) como um substituto robusto para o treinamento do modelo.
    """
    # LINK SUBSTITUTO (Estável): UCI Air Quality Dataset hospedado no GitHub
    url = "https://raw.githubusercontent.com/asharvi1/UCI-Air-Quality-Data/master/AirQualityUCI.csv"
    _download_file(url, "poluicao_uci_historico.csv", encoding='utf-8')

def main():
    """
    Função principal para orquestrar a coleta de todos os dados.
    """
    # Garante que o diretório 'data/raw' exista
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    
    print("--- Iniciando Coleta de Dados (Versão Corrigida) ---")
    
    # 1. Dados de Autuações (Feature Histórica)
    fetch_ibama_data()
    
    # 2. Dados Geoespaciais (Feature de Proximidade)
    fetch_unidades_conservacao()
    
    # 3. Dados de Poluição (Feature e Tempo Real)
    fetch_poluicao_data()
    
    print("--- Coleta de Dados Concluída ---")

if __name__ == "__main__":
    main()