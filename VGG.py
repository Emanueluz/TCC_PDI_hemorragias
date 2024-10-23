import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import progressbar

def get_classes_from_csv(csv_file):
    # Ler o arquivo CSV
    df = pd.read_csv(csv_file, sep=';')
    classes = df.columns[1:].tolist()  # Ajuste o índice se necessário
    return classes

# Definir o caminho para o diretório de imagens e o arquivo CSV de rótulos
IMAGE_DIR = "/hd1/rsna/png_pre_pro/"
TRAIN_CSV = '/home/emanuel/Desktop/TCC_PDI_hemorragias-main/train2.csv'
VAL_CSV = "/home/emanuel/Desktop/TCC_PDI_hemorragias-main/val2.csv"
TEST_CSV = "/home/emanuel/Desktop/TCC_PDI_hemorragias-main/test2.csv"

# Hiperparâmetros
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_CLASSES = 6  # Número de classes, ajustado conforme o número de colunas no CSV
PATIENCE = 5
# Definir transformações de dados
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Criar uma classe de Dataset personalizado
class CustomMultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file, sep=',')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path + ".png").convert("RGB")
        labels = self.img_labels.iloc[idx, 1:].astype(float).values

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)

# Carregar os conjuntos de dados de treino, validação e teste
train_dataset = CustomMultiLabelDataset(csv_file=TRAIN_CSV, img_dir=IMAGE_DIR, transform=transform)
val_dataset = CustomMultiLabelDataset(csv_file=VAL_CSV, img_dir=IMAGE_DIR, transform=transform)
test_dataset = CustomMultiLabelDataset(csv_file=TEST_CSV, img_dir=IMAGE_DIR, transform=transform)

# Criar DataLoaders para treinamento, validação e teste
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True)

# Carregar o modelo ResNet pré-treinado
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)

# Definir o dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")
model.to(device)

# Definir a função de perda e o otimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Criar diretório de relatórios
reports_dir = "resultados"
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

# Variável para armazenar a melhor acurácia de validação
best_val_accuracy = 0.0
import pandas as pd



# Loop de treinamento
best_val_accuracy = 0.0
i=0
counter=0
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    bar = progressbar.ProgressBar(max_value=len(train_loader))
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        bar.update(i)
        i+=1
        optimizer.zero_grad()


        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    i=0
    epoch_duration = time.time() - epoch_start_time
    avg_loss = running_loss / len(train_loader)

    # Validação
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.numel()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Duration: {epoch_duration:.2f} seconds")
    cl=get_classes_from_csv(TRAIN_CSV)
    # Salvar métricas e matriz de confusão


    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        best_model_path = os.path.join(reports_dir, f"resnet-best_model.pth")
        torch.save(model, best_model_path)
        print(f"Melhor modelo salvo com acurácia de validação: {accuracy:.2f}%")
        counter=0
    else:
        counter += 1  # Aumenta o contador de paciência
        if counter >= PATIENCE:
            print(f"Early stopping na época {epoch + 1}. A perda de validação não melhorou após {PATIENCE} épocas.")
            break
print("Treinamento concluído.")

# Teste
model.eval()
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        correct += (predicted == labels).sum().item()
        total += labels.numel()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

 
