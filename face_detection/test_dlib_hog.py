import dlib
import os
import cv2
import numpy as np
import time
import scipy.stats as st


hog_face_detector = dlib.get_frontal_face_detector()

def detect(image, filename):
    # Transforma imagem de BGR para RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Aplica o algoritmo de detecção de faces, que devolve as bounding boxes e confiança.
    t0 = time.time()
    boxes, confidences, matches = hog_face_detector.run(image, 1, 1)
    i=0
    with open(("results_dlib_hog/"+filename.split(".jpg")[0])+".txt", 'w') as f:
        for i, box in enumerate(boxes):
            cv2.rectangle(image, (int(box.left()), int(box.top())), (int(box.right()), int(box.bottom())), (0, 255, 0), 2)
            f_line = "face " + str(st.norm.cdf(confidences[i])) + " " +str(box.left())+" " +str(box.top())+" "+str(box.right())+" "+str(box.bottom())+"\n"
            i+=1
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

    import dlib
    print(dlib.DLIB_USE_CUDA)
    dataset_folder = 'dataset/WIDER_val/images/'
    list_f = os.listdir(dataset_folder)

    t0 = time.time()
    for foldername in list_f:
        for filename in os.listdir(dataset_folder+foldername):
            img = cv2.imread(dataset_folder+foldername+"/"+filename)
            if img is not None:
                detect(img, filename)
    print("Tempo: {}".format(time.time()-t0))