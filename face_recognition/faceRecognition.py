# Imports padroes
import pickle
import os
import sys
import errno
import time


# Imports visão computacional
import cv2
import face_recognition
import imutils
import numpy as np


##  @brief Classe para realizar a detecção de rostos.
class FaceDetection:

    ##  @brief Construtor da classe
    #   @param model_path Caminho para arquivo com o modelo treinado do Ultra_light_face_detector.
    #   @param image_height Altura das imagens de entrada.
    #   @param image_width Largura das imagens de entrada.
    #   @param min_confidence Limiar para exclusão de boxes com baixa confiança.
    def __init__(self, model_path, image_height, image_width, min_confidence = 0.6):

        # Adiciona caminho para codigo externo

        # Imports para carregamento do modelo de detector
        from vision.ssd.config.fd_config import define_img_size
        define_img_size(image_width)
        from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

        # Cria as estruturas do modelo
        test_device = "cuda:0"
        self.net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=test_device)
        self.predictor = create_Mb_Tiny_RFB_fd_predictor(self.net, candidate_size=1500, device=test_device)
    
        # Carrega o modelo
        self.net.load(model_path)

        ## Tamanho da imagem do modelo
        self.image_height = image_height
        self.image_width = image_width

        ## Parametros para thresholg
        self.min_confidence = min_confidence


    ##  @brief Função que coloca as imagens no tamanho correto do preditor sem alterar seu aspecto,
    #          preenche com preto quando a imagem é desproporcional ao tamanho do modelo.
    #   @param image Imagem que será alterada.
    #   @return Imagem alterada.
    def __imageResize(self, image):

        # Armarzena tamanho original da imagem
        image_height, image_width, image_channels = image.shape
        
        w_ratio = float(image_width/self.image_width)
        h_ratio = float(image_height/self.image_height)

        # Cria imagem preta do tamanho correto
        resized_image = np.zeros(shape=(self.image_height, self.image_width, image_channels), dtype=np.uint8)

        # Coloca a imagem desejada no centro da imagem preta
        if(h_ratio > w_ratio):
            image = imutils.resize(image, height=self.image_height)
            w_offset = int(self.image_width/2-image.shape[1]/2)
            resized_image[0:self.image_height, w_offset:w_offset+image.shape[1]] = image
        else:
            image = imutils.resize(image, width=self.image_width)
            h_offset = int(self.image_height/2-image.shape[0]/2)
            resized_image[h_offset:h_offset+image.shape[0], 0:self.image_width] = image

        return resized_image


    ##  @brief Realiza pré-processamento nas imagens.
    #   @param image Imagem que será alterada.
    #   @return Imagem alterada.
    def __preprocessImage(self, image):

        # Coloca imagem no tamanho correto do modelo
        image = self.__imageResize(image)
        # Faz transformação de BGR para RGB.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, rgb_image


    ##  @brief Funcão que realiza detecção.
    #   @param image Imagem para procurar os rostos.
    #   @return Imagem alterada pelo modelo e bounding boxes de rostos encontrados.
    def detect(self, image):

        # Faz pré-processamento da imagem
        image, rgb_image = self.__preprocessImage(image)
        boxes, _, confidences = self.predictor.predict(rgb_image, 800, self.min_confidence)

        if(len(boxes)>0):
            boxes = boxes[:, [1,2,3,0]]

        return image, boxes


##  @brief Classe que faz reconhecimento de rostos baseada em arquivos de registro.
class FaceRecognition:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []

##  @brief Carrega faces conhecidas.
    #   @param filename Nome do arquivo.
    def loadRegister(self, filename=None):  

        # Verifica se o arquivo existe. 
        try:     
            if os.path.isfile(filename):
                    # with open(filename, "rb") as f:
                    f = open(filename, "rb")
                    data = pickle.loads(f.read())
                    self.known_encodings = data["encodings"]
                    self.known_names = data["names"]
            else:
                f = open(filename, 'wb')
                f.close()
                self.known_encodings = []
                self.known_names = []
                self.save_register(filename)
                

        except Exception as exp:
            print(exp)
            self.known_encodings = []
            self.known_names = []
           

    ##  @brief Registra face nos arquivos de face conhecida.
    #   @param image Imagem com face a ser cadastrada.
    #   @param boxes Boxes de rostos encontrados.
    #   @param name Nome da pessoa presente na foto.
    def register(self, image, boxes, name):
        ## Encontra as features de rosto dentro das boxes
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Passa por todos os encoadings e salva da lista de pessoas conhecidadas
        for encoding in encodings:
            self.known_encodings.append(encoding)
            self.known_names.append(name)

    ##  @brief Salva o registro de faces conhecidas
    #   @param filename Nome do arquivo para salvar
    def save_register(self, filename):
        try:
            data = {"encodings": self.known_encodings, "names": self.known_names}
            f = open(filename, "wb")
            f.write(pickle.dumps(data))
            f.close()
        except:
            raise ChildProcessError(errno.ECHILD, os.strerror(errno.ECHILD))


    def delete_object(self, name):

        for i in range(len(self.known_names)-1,-1,-1):
            if self.known_names[i]==name:
                del self.known_encodings[i]
                del self.known_names[i]

    
        
    ##  @brief Reconhece rosto em lista de rostos conhecidos.
    #   @param image Imagem com rosto a ser cadastrado.
    #   @param boxes Boxes de rostos encontrados.
    #   @param name Nome da pesso presente na foto.
    def recognize(self, image, boxes, draw_in_image=False, tolerance=0.5, min_recognition = 5):

        names=[]

        ## Encontra as features de rosto dentro das boxes
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            name = "Desconhecido"
            ## Compara rostos da foto com os do registro.
            matches = face_recognition.compare_faces(self.known_encodings,encoding,tolerance=tolerance)
            matches_d = face_recognition.face_distance(self.known_encodings,encoding)

            ## Se houver correspondencia, salva o nome da pessoa presente na foto.
            if True in matches: 
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = self.known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                if counts[max(counts, key=counts.get)]>=min_recognition:
                    name = max(counts, key=counts.get)
                    names.append(name)

            if draw_in_image:
                for i in range(len(boxes)):
                    y1, x2, y2, x1 = boxes[i]
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),2)
                    cv2.putText(image, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        

        return image, names


