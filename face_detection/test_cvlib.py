## @file test_cvlib.py
#  
# Esse arquivo contém o código de teste de desempenho e qualidade da biblioteca cvlib.

import cvlib
import os
import cv2
import numpy as np
import time



def detect(image, filename):
    # Transforma imagem de BGR para RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Aplica o algoritmo de detecção de faces, que devolve as bounding boxes e confiança.
    boxes, confidences = cvlib.detect_face(rgb)
    i=0
    if len(boxes)>0:
        with open(("results_cvlib/"+filename.split(".jpg")[0])+".txt", 'w') as f:
            for (left, top, right, bottom ) in boxes:
                cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                f_line = "face " +str(confidences[i])+ " " +str(left)+" " +str(top)+" "+str(right)+" "+str(bottom)+"\n"
                i+=1
                f.write(f_line)

    if filename == '38_Tennis_Tennis_38_371.jpg':
        cv2.imshow("image", image)
        cv2.waitKey(0)


#   Script para testar o código de avaliação da biblioteca cvlib.
if __name__ == '__main__':
    t0 = time.time()
    dataset_folder = 'dataset/WIDER_val/images/'
    list_f = os.listdir(dataset_folder)

    for foldername in list_f:
        for filename in os.listdir(dataset_folder+foldername):
            img = cv2.imread(dataset_folder+foldername+"/"+filename)
            if img is not None:
                detect(img, filename)
        
    print("Tempo: {}".format(time.time()-t0))