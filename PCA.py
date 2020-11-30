import os
import sys
import numpy as np
import random
import cv2


class DataSet():


    def __init__(self, path, porcentagem_treino):    
        self.path = path
        self.porcentagem_treino = porcentagem_treino * 10
        

    def get_imagem(self, arquivo):
    
        img = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(img, (80, 80))
        matriz = resized.T.reshape((1, resized.shape[1] * resized.shape[0]))
        return np.float64(matriz)


    def load_dataset(self, numero_exemplos):
    
        dataset_treino, dataset_teste = [], []
        for root, _, files in os.walk(self.path):
            images = [os.path.join(root, file) for file in files if file.endswith(".jpg")]
        dataSet = []

        for i in images:
            aux = i[i.rfind("\\") + 1 : i.rfind(".jpg")]
            data = aux.split("_")
            dataFile = self.get_imagem(i)
            p = Person(int(data[0]), int(data[1]), dataFile)
            dataSet.append(p)
  
        dataSet.sort(key=lambda p: p.id)

        index = 0
        
        while index < len(dataSet):
            exemplos = dataSet[index: index + numero_exemplos]

            while len(exemplos) > self.porcentagem_treino:
                i = random.randint(0, len(exemplos) - 1)
                dataset_teste.append(exemplos.pop(i))

            if self.porcentagem_treino == numero_exemplos:
                dataset_teste.extend(exemplos)    

            dataset_treino.extend(exemplos)
            index += numero_exemplos

        return (dataset_treino, dataset_teste)     
        

class Person():

    def __init__(self, id, label, data):
        self.id = id
        self.label = label
        self.data = data

dataset = DataSet('dataset', 0.7)
base_treino = []
base_teste = []
base_treino, base_teste = dataset.load_dataset(10)

for i in range(10, 21):
    
    model = cv2.face.EigenFaceRecognizer_create(i)
    im = []
    labels = []
    
    for img in base_treino:
        im.append(img.data)
        labels.append(img.label)

    model.train(im, np.asarray(labels))

    j = 0
    
    for img in base_teste:
        label, confianca = model.predict(img.data)

        if img.label == label:
            j += 1
                    

    taxa_acerto = j/len(base_teste) * 100
    print('_' * 25)
    print('PCA: ', i, ' componentes')
    print('Taxa de acerto: ', taxa_acerto, '%')

print('Treino: ', len(base_treino), ' imagens')
print('Teste: ', len(base_teste),' imagens')