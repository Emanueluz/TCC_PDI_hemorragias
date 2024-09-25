import os
import pandas as pd
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import progressbar
import time
# Definir o caminho para o diretório de imagens e o arquivo CSV de rótulos
IMAGE_DIR = "/home/emanuel/Imagens/dataset_pre_pro/"
TRAIN_CSV = '/home/emanuel/Documentos/tcc/genesis-brain-hemorrhage-main/data/rsna/data_csv_files/tiny_files/train.csv'
VAL_CSV = "/home/emanuel/Documentos/tcc/genesis-brain-hemorrhage-main/data/rsna/data_csv_files/tiny_files/val.csv"
TEST_CSV = "/home/emanuel/Documentos/tcc/genesis-brain-hemorrhage-main/data/rsna/data_csv_files/tiny_files/test.csv"

# Hiperparâmetros
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_CLASSES = 6  # Número de classes, ajustado conforme o número de colunas no CSV
# Definir transformações de dados
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# Carregar os conjuntos de dados de treino, validação e teste
train_dataset = CustomMultiLabelDataset(csv_file=TRAIN_CSV, img_dir=IMAGE_DIR, transform=transform)
val_dataset = CustomMultiLabelDataset(csv_file=VAL_CSV, img_dir=IMAGE_DIR, transform=transform)
test_dataset = CustomMultiLabelDataset(csv_file=TEST_CSV, img_dir=IMAGE_DIR, transform=transform)

# Criar DataLoaders para treinamento, validação e teste
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Carregar o modelo Vision Transformer (ViT) pré-treinado
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
# Ajustar o cabeçalho final para o número de classes
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)

# Definir o dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir a função de perda para classificação multirrótulo
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Criar o diretório para salvar os resultados
results_dir = "resultados"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Variável para armazenar a melhor acurácia de validação
best_val_accuracy = 0.0
i=0
counter=0
# Loop de treinamento
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()  # Marca o início do tempo da época
    model.train()
    running_loss = 0.0
    bar = progressbar.ProgressBar(max_value=len(train_loader))

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        i+=1
        time.sleep(0.1)  # Substitua isso pelo seu processamento
        bar.update(i)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    i=0
    epoch_duration = time.time() - epoch_start_time  # Calcula a duração da época
    avg_loss = running_loss / len(train_loader)

    # Validação
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Duration: {epoch_duration:.2f} seconds")

    # Salvar o melhor modelo
    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        best_model_path = os.path.join(results_dir, f"ViT-best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Melhor modelo salvo com acurácia de validação: {accuracy:.2f}%")

print("Treinamento concluído.")

# Teste
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.numel()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
