import csv

 
def ler_arquivo_de_csv_original(arquivo):
    lista=[]
    exame=[]
    q=0
    with open(arquivo, newline="") as csv_original:
        arquivo_de_texto=csv.reader(csv_original)
        
        for linha in arquivo_de_texto:
            q+=1
            if (len(linha[0].split("_"))>2):
                if(len(exame)==7):
                    lista.append(exame)
                    exame=[]    
                if (exame==[]):
                    exame.append((linha[0].split("_"))[1])
                    exame.append(linha[1])

                else:
                    exame.append(linha[1])
        lista.append(exame)

     
    return lista



def comparar_listas(lista1,lista2):
    resultado=[]
    j=0;i=0
    while(i < len(lista1)-1): #lista com menos loops
        encontrar=False
        while(encontrar==False and j<len(lista2)-1):#lista com todos os loops
            #print(lista1[i][0],"=====",lista2[j][0])
            if lista1[i][0]==lista2[j][0]:
                resultado.append(lista2[j])
                lista2.pop(j)
                encontrar=True
                print("encontrado")
            j+=1

        
        i+=1
    print(resultado)
    return resultado


def escrever_csv(lista, nome_do_arquivo):
    arquivo= open(nome_do_arquivo+".csv","w")
    a = csv.writer(arquivo)
    a.writerow(["ID","epidural","intraparenchymal","intraventricular","subarachnoid","subdural","any"])
    a.writerows(lista)
    arquivo.close()
    return 


def separador_de_positivos(lista):
    

    return lista_de_tipo_de_hemorragias

def main():
    lista=ler_arquivo_de_csv_original("/home/emanuel/Área de Trabalho/stage_2_train.csv")
    lista_treino=ler_arquivo_de_csv_original("/media/emanuel/809AEADC9AEACE2A/TCC/stage_2_sample_submission.csv")
    #comparar_listas(lista_treino, lista)
    escrever_csv(lista, "/home/emanuel/Área de Trabalho/TCC/novo_arquivo_de_train")
    
    
    return 0

main()


 