import time
import os
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

from faceRecognition import FaceDetection
from faceRecognition import FaceRecognition

model_path = 'models/pretrained/version-RFB-640.pth'
register_path = 'registro.pickle'
h = 480
w = 640
fd = FaceDetection(model_path, h, w)
fr = FaceRecognition()
fr.loadRegister(register_path)



def register(filenames):
    for filename in filenames:
        image = cv2.imread(filename)

        if image is not None:
            name = filename.split('/')[2]

            image, boxes = fd.detect(image)
            if len(boxes>0):
                fr.register(image, boxes, name)

    fr.save_register(register_path)

def plot_confusion_matrix(y_true, y_pred, classes, title):
    acc = accuracy_score(y_true, y_pred)
    title = title + " (Acurácia Total:" + str("{:10.4f}".format(acc)) + ")"

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    print(cm)
    # cm_df = pd.DataFrame(cm, index = classes, columns = classes)
    # fig = plt.figure(figsize=(5.5,4))
    # sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
    # plt.title(title)
    # plt.ylabel('Classe Verdadeira')
    # plt.xlabel('Classe Predita')
    # fig.tight_layout()
    # plt.show()

def inference(filenames, classes):
    gt = []
    inference = []

    t0 = time.time()
    for filename in filenames:
        image = cv2.imread(filename)
        real_name = filename.split('/')[2]
        if image is not None:
            gt.append(real_name)
            image, boxes = fd.detect(image)
            image, names = fr.recognize(image, boxes, min_recognition=2)
            if(len(names))>0:
                inference.append(names[0])
            else:
                inference.append("Desconhecidos")
        else:
            print("Problema com imagem ", filename)
        
    print(time.time()-t0)
    plot_confusion_matrix(gt, inference, classes, "Matriz de Confusão das Inferências de Reconhecimento Facial")



#   Script para testar o código de avaliação da biblioteca cvlib.
if __name__ == '__main__':
    t0 = time.time()
    dataset_folder = 'dataset/img/'
    list_f = os.listdir(dataset_folder)
    files = []
    N = 35
    start = 15
    classes = []

    for foldername in list_f:
        if foldername == 'Desconhecidos':
            continue
        classes.append(foldername)
        i = 0
        for filename in os.listdir(dataset_folder+foldername):
            i+=1
            if i >= start:
                files.append(dataset_folder+foldername+'/'+filename)
            if i==start+N:
                break

    classes.append("Desconhecidos")
    for filename in os.listdir(dataset_folder+'Desconhecidos'):
        files.append(dataset_folder+'Desconhecidos'+'/'+filename)
    

    print(len(files))

    random.shuffle(files)
    inference(files, classes)
    
    
        
