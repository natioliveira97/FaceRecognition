## @file test_ulfd2.py
#  
# Esse arquivo contém o código de teste de desempenho e qualidade do modelo Ultra Light Fast Generic Face Detector
import os
import cv2
import numpy as np
import time



# Configura tamanho das imagens
from vision.ssd.config.fd_config import define_img_size
define_img_size(640)

# Importa estrutura do modelo
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
test_device = "cuda:0"
net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=test_device)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1500, device=test_device)

# Carrega o modelo
model_path = "models/pretrained/version-RFB-640.pth"
net.load(model_path)


def detect(image, filename, show_results=False):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Aplica o algoritmo de detecção de faces, que devolve as bounding boxes e confiança.
    boxes, _, confidences = predictor.predict(image, 1500 / 2, 0.6)

    if(len(boxes)>0):
        with open(("results_ulfd2/"+filename.split(".jpg")[0])+".txt", 'w') as f:
            for i, d in enumerate(boxes):
                x1, y1, x2, y2 = d.tolist()
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                f_line = "face " + str(float((confidences[i]))) + " " +str(int(x1))+" " +str(int(y1))+" "+str(int(x2))+" "+str(int(y2))+"\n"
                f.write(f_line)

            if filename == '38_Tennis_Tennis_38_371.jpg':
                cv2.imshow("image", image)
                cv2.waitKey(0)


if __name__ == '__main__':

    f = 0
    t0 = time.time()
    dataset_folder = 'dataset/WIDER_val/images/'
    list_f = os.listdir(dataset_folder)

    for foldername in list_f:
        for filename in os.listdir(dataset_folder+foldername):
            img = cv2.imread(dataset_folder+foldername+"/"+filename)
            if img is not None:
                detect(img, filename)
                f+=1
    print(f)
    print("Tempo: {}".format(time.time()-t0))

