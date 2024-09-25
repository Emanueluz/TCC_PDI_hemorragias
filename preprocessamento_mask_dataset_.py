import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.dicomdir import DicomDir
import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler
from scipy.signal import find_peaks
pydicom.config.image_handlers = [None, gdcm_handler]
import tqdm



def biggest_component(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    max_label = 1
    if len(sizes) > 1:
        max_size = sizes[1]
    else:
        max_size = sizes[0]  
    
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    mask = np.zeros(output.shape)
    mask[output == max_label] = 255
    return mask


 
def window_image(img: np.ndarray,
                  window_center: int,
                  window_width: int,
                  rescale: bool = True) -> np.ndarray:

    img = img.astype(np.float32)
    # for translation adjustments given in the dicom file.
    img_min = window_center - window_width//2  # minimum HU level
    img_max = window_center + window_width//2  # maximum HU level
    # set img_min for all HU levels less than minimum HU level
    img[img < img_min] = img_min
    # set img_max for all HU levels higher than maximum HU level
    img[img > img_max] = img_max
    if rescale:
        img = (img - img_min) / (img_max - img_min)*255.0
    return img


def listar_arquivos(diretorio):
    """
    Lista todos os arquivos em um diretório.

    Parâmetros:
        diretorio (str): Caminho para o diretório.

    Retorna:
        list: Lista de nomes de arquivos no diretório.
    """
    try:
        # Verifica se o caminho fornecido é um diretório
        if os.path.isdir(diretorio):
            # Lista todos os arquivos no diretório
            arquivos = [f for f in os.listdir(diretorio) if os.path.isfile(os.path.join(diretorio, f))]
            return arquivos
        else:
            return f"Erro: {diretorio} não é um diretório válido."
    except Exception as e:
        return f"Erro ao listar arquivos: {e}"

 



def pre_process_img(caminho_arquivo):    
    data = pydicom.dcmread(caminho_arquivo)
    img_raw = data.pixel_array.astype(np.float64)
    img_raw = img_raw * data.RescaleSlope + data.RescaleIntercept


    img = window_image(img_raw, 80, 120)

    img = img.astype(np.uint8)
    img = cv2.resize(img, (256, 256))
    
    mask = np.zeros_like(img) + 255

    mask[img < 120] = 0
    
    masked_img = np.copy(img)
    masked_img[mask == 255] = 0
    

    brain_mask = biggest_component(masked_img)
    
    masked_brain = np.copy(masked_img)
    masked_brain[brain_mask == 0] = 0
    
    return masked_brain


def salvar_imagem_cv2(imagem, diretorio, nome_arquivo):
    
    # Verifica se o diretório existe, se não, cria
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
    # Define o caminho completo do arquivo
    caminho_completo = os.path.join(diretorio, nome_arquivo)
    # Salva a imagem
    cv2.imwrite(caminho_completo, imagem)
    return f"Imagem salva com sucesso em: {caminho_completo}"





def main():

    diretorio__dataset_original= "/home/emanuel/Imagens/dicom/"
    diretorio__dataset_destino = "/home/emanuel/Imagens/dataset_pre_pro/" 
    try:
        # Verifica se o diretório já existe
        if not os.path.exists(diretorio__dataset_destino):
            # Cria o diretório
            os.makedirs(diretorio__dataset_destino)
            print(f"Diretório '{diretorio__dataset_destino}' criado com sucesso.")
        else:
            print(f"Diretório '{diretorio__dataset_destino}' já existe.")
    except Exception as e:
        return f"Erro ao criar o diretório: {e}"


    
    arquivos = listar_arquivos(diretorio__dataset_original)

    for img in tqdm.tqdm(arquivos):
        imagem =  pre_process_img(diretorio__dataset_original+img)
        salvar_imagem_cv2(imagem,diretorio__dataset_destino, img[:-3]+"png" )



    return 0      


main()