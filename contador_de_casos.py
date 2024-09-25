import pandas as pd

def somar_colunas_csv(caminho_csv):
    # Ler o arquivo CSV
    df = pd.read_csv(caminho_csv, sep=';')
    
    # Calcular a soma de cada coluna
    soma_colunas = df.sum(numeric_only=True)  # Ignora colunas não numéricas
    
    # Imprimir os resultados
    print("Soma de cada coluna:")
    print(soma_colunas)
    
    return soma_colunas

# Exemplo de uso
caminho_csv = '/home/emanuel/Documentos/tcc/genesis-brain-hemorrhage-main/data/rsna/data_csv_files/tiny_files/train.csv'  # Substitua pelo caminho do seu arquivo CSV
somar_colunas_csv(caminho_csv)
