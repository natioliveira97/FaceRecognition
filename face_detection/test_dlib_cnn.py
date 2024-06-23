import dlib
import os
import cv2
import numpy as np
import time
import scipy.stats as st


cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

def detect(image, filename):
    # Transforma imagem de BGR para RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Aplica o algoritmo de detecção de faces, que devolve as bounding boxes e confiança.
    dets = cnn_face_detector(image, 1)
    
    with open(("results_dlib_cnn/"+filename.split(".jpg")[0])+".txt", 'w') as f:
        for i, d in enumerate(dets):
            cv2.rectangle(image, (int(d.rect.left()), int(d.rect.top())), (int(d.rect.right()), int(d.rect.bottom())), (0, 255, 0), 2)
            if d.confidence>1:
                d.confidence = 1.0

            f_line = "face " + str((d.confidence)) + " " +str(d.rect.left())+" " +str(d.rect.top())+" "+str(d.rect.right())+" "+str(d.rect.bottom())+"\n"
            f.write(f_line)
        if filename == '38_Tennis_Tennis_38_371.jpg':
                cv2.imshow("image", image)
                cv2.waitKey(0)


# def test(image show_results = False):
#     # Inicializa o tempo de detecção com zero.
#     self.detection_time=0
#     for image in images:
#         # Guarda momento que começa o processamento da imagem
#         t1=time.time()
#         # Transforma imagem de BGR para RGB
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # Aplica o algoritmo de detecção de faces, que devolve as bounding boxes eencontradas.
#         boxes = face_recognition.face_locations(rgb,model=self.method)
#         # Calcula o tempo que levou para processar uma imagem e adiciona ao tempo de detecação.
#         self.detection_time += time.time()-t1
#         if show_results:
#             for (top, right, bottom, left) in boxes:
#                 cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
#             cv2.imshow("Image", image)
#             cv2.waitKey(0)


#   Script para testar o código de avaliação da biblioteca dlib
if __name__ == '__main__':
    t0 = time.time()
    print(dlib.DLIB_USE_CUDA)
    dataset_folder = 'dataset/WIDER_val/images/'
    list_f = os.listdir(dataset_folder)

    for foldername in list_f:
        for filename in os.listdir(dataset_folder+foldername):
            img = cv2.imread(dataset_folder+foldername+"/"+filename)
            if img is not None:
                detect(img, filename)

    print("Tempo: {}".format(time.time()-t0))