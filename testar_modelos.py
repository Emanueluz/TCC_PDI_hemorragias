import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import time
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
# Definir o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Criar uma classe de Dataset personalizado
class CustomMultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file, sep=';')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path + ".png").convert("RGB")
        
        # Pega os labels das colunas de 1 até o final
        labels = self.img_labels.iloc[idx, 1:].astype(float).values

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)
# Função para carregar o modelo salvo
def carregar_modelo(caminho_pth):
    modelo = torch.load(caminho_pth)
    modelo.to(device)
    modelo.eval()  # Colocar o modelo em modo de avaliação
    return modelo

# Função para avaliar o modelo
def avaliar_modelo(modelo, dataloader):
    todos_rotulos = []
    todas_predições = []
    
    with torch.no_grad():
        
        for dados, rotulos in dataloader:
            dados, rotulos = dados.to(device), rotulos.to(device)
            saídas = modelo(dados)
            previsões = (torch.sigmoid(saídas) > 0.5).float()

            todos_rotulos.extend(rotulos.cpu().numpy())
            todas_predições.extend(previsões.cpu().numpy())

    return np.array(todos_rotulos), np.array(todas_predições)

# Função para calcular as métricas e salvar em CSV
def calcular_metricas_e_salvar(rotulos, predições, nome_arquivo_csv):
    # Calcular as métricas
    acuracia = accuracy_score(rotulos, predições)
    precisao = precision_score(rotulos, predições, average='macro')
    revocacao = recall_score(rotulos, predições, average='macro')
    f1 = f1_score(rotulos, predições, average='macro')
    classes= ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']
    # Matriz de confusão

    # Criar um DataFrame com as métricas
    resultados = {
        'Acurácia': [acuracia],
        'Precisão (macro)': [precisao],
        'Revocação (macro)': [revocacao],
        'F1-Score (macro)': [f1],
    }

    # Salvar métricas em CSV
    df_metricas = pd.DataFrame(resultados)
    df_metricas.to_csv(nome_arquivo_csv, index=False)
    matriz_confusao = multilabel_confusion_matrix(rotulos, predições)
    
    #Inicializa a matriz de confusão global
    matriz_confusao_global = np.sum(matriz_confusao, axis=0)

    # Calcular a matriz de confusão para cada classe
    for i in matriz_confusao:
         matriz_confusao_global+=i
    
    cconfusao = unique_labels(rotulos, predições)
    print(matriz_confusao.shape)
    title='Matriz de Confusão'
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao_global, annot=True, fmt='d', cmap='Blues',xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Verdadeira')
    plt.tight_layout()
    caminho_arquivo = os.path.join( f"matriz_confusao_classe_{'teste'}.png")
    plt.savefig(caminho_arquivo)
    plt.close()

    # Salvar matriz de confusão em CSV
    for i, matrix in enumerate(matriz_confusao):
        plt.figure(figsize=(5, 5))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Previsão')
        plt.ylabel('Verdadeiro')
        plt.title(f'Matriz de Confusão para a Classe: {classes[i]}')

        # Salvar imagem em formato PNG
        caminho_arquivo = os.path.join( f"matriz_confusao_classe_{classes[i]}.png")
        #plt.savefig(caminho_arquivo)
        plt.close()  # Fechar a figura para liberar memória
    
    print(f"Métricas e matriz de confusão salvas em: {nome_arquivo_csv}")


def listar_arquivos(diretorio):
    # Verifica se o caminho é um diretório válido
    if os.path.isdir(diretorio):
        # Lista todos os arquivos e diretórios no caminho fornecido
        arquivos = os.listdir(diretorio)
    return arquivos

# Definir as transformações de teste
transformacao_teste = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Definir o caminho para o diretório de imagens e o arquivo CSV de rotulos
IMAGE_DIR = "/home/emanuel/Imagens/dataset_pre_pro/"
TEST_CSV = "/home/emanuel/Documentos/tcc/genesis-brain-hemorrhage-main/data/rsna/data_csv_files/tiny_files/test.csv"
test_dataset = CustomMultiLabelDataset(csv_file=TEST_CSV, img_dir=IMAGE_DIR, transform=transformacao_teste)
di_modelos ='/home/emanuel/Documentos/tcc/resultados'
lista_de_modelos=listar_arquivos(di_modelos)
 

# Carregar o conjunto de dados de teste
#dataset_teste = datasets.ImageFolder(root='/home/emanuel/Documentos/tcc/genesis-brain-hemorrhage-main/data/rsna/data_csv_files/tiny_files/test.csv', transform=transformacao_teste)
#dataloader_teste = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
for nome_modelo in lista_de_modelos:
    # Caminho para o modelo salvo (.pth)
    caminho_modelo = di_modelos+'/'+nome_modelo

    # Carregar o modelo
    modelo = carregar_modelo(caminho_modelo)

    # Avaliar o modelo no conjunto de teste
    rotulos, predições = avaliar_modelo(modelo, test_loader)

    # Calcular as métricas e salvar os resultados em CSV
    nome_arquivo_csv = 'resultados_metricas'+nome_modelo+'.csv'
    calcular_metricas_e_salvar(rotulos, predições, nome_arquivo_csv)
