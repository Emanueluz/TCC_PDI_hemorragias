import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

 
 
'''
# Configura o TensorFlow para usar a primeira GPU (ID 0)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Define a GPU específica que você deseja usar
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
    except RuntimeError as e:
        print(e)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
'''

def listar_arquivos(diretorio):
    # Verifica se o caminho é um diretório válido
    if os.path.isdir(diretorio):
        # Lista todos os arquivos e diretórios no caminho fornecido
        arquivos = os.listdir(diretorio)
    return arquivos

def carregar_imagem(img_id, image_dir):
    img_path = os.path.join(image_dir, img_id)
    img = tf.io.read_file(img_path)
    return img

# Função para processar os dados de treino/validação/teste
def processar_dados(df, image_dir):
    imagens = []
    labels = []
    img_size=(256, 256)
    for index, row in df.iterrows():
        img_id = row['ID'] + ".png"
        img = carregar_imagem(img_id, image_dir)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)  # Redimensiona a imagem
        img = tf.cast(img, tf.float32) / 255.0
        label = np.array(row[1:].values, dtype=np.float32)
        imagens.append(img)
        labels.append(label)
    image_ds = list_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = tf.data.Dataset.from_tensor_slices((imagens, labels)).cache()
    dataset = dataset.batch(16)
    return dataset #tf.convert_to_tensor(imagens, dtype=tf.float32).bath, tf.convert_to_tensor(labels)

def dividir_lista(lista, tamanho_sublista):
    # Usa list comprehension para dividir a lista original em sublistas de tamanho fixo
    return [lista[i:i + tamanho_sublista] for i in range(0, len(lista), tamanho_sublista)]



def main():
    modelos_sem_treino = "/home/emanuel/Documentos/tcc/modelos_sem_treino"
    lista_de_modelos=listar_arquivos(modelos_sem_treino)
    modelos_treinados= "modelos_treinados"
    IMAGE_DIR = "/home/emanuel/Imagens/dataset_pre_pro/"
    TRAIN_CSV = '/home/emanuel/Documentos/tcc/genesis-brain-hemorrhage-main/data/rsna/data_csv_files/tiny_files/train.csv'
    VAL_CSV = '/home/emanuel/Documentos/tcc/genesis-brain-hemorrhage-main/data/rsna/data_csv_files/tiny_files/val.csv'
    TEST_CSV =  '/home/emanuel/Documentos/tcc/genesis-brain-hemorrhage-main/data/rsna/data_csv_files/tiny_files/test.csv'
    batch_size = 36
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    NUM_CLASSES = 6
    train_df = pd.read_csv(TRAIN_CSV, delimiter=';')
    val_df = pd.read_csv(VAL_CSV, delimiter=';')
    test_df = pd.read_csv(TEST_CSV, delimiter=';')

    lista_dataset= dividir_lista(train_df,1000)

    #x_train, y_train = processar_dados(train_df, IMAGE_DIR)
    dataset_val = processar_dados(val_df, IMAGE_DIR)
    #x_test, y_test = processar_dados(test_df, IMAGE_DIR)
    num_classes=6
    input_shape=(256,256,3)

    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Congelar as camadas do modelo base inicialmente

    # Construindo o modelo completo
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compilando o modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinando o modelo
    for dataset in lista_dataset:
        dataset = processar_dados(train_df, IMAGE_DIR)

        history = model.fit(dataset, #x_train, y_train,  # x_train e y_train devem estar definidos
                    epochs=NUM_EPOCHS,
                    validation_data=(dataset_val),
                    verbose=1
                    )
   
    # Descongelando as camadas da ResNet50 para afinar o modelo
    base_model.trainable = True

    # Recompilando o modelo com um novo otimizador (taxa de aprendizado menor)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate / 10)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Continuando o treinamento com camadas descongeladas
    history_fine_tune = model.fit(x_train, y_train,  # x_train e y_train devem estar definidos
                    batch_size=batch_size,
                    epochs=NUM_EPOCHS,
                    validation_data=(x_val, y_val),
                    verbose=1
                    )
   


main()


