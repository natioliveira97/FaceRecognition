import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import os
import cv2
import numpy as np
import time

##  @brief Classe que realiza os testes e avaliação do modelo Ultra Light Fast Generic Face Detector.
class TestUltraLightFaceDetector:
    ##  @brief Construtor da classe 
    ##  @param onnx_path caminho para o arquivo onnx com o modelo já treinado.
    def __init__(self, onnx_path):
        ## Tempo de execução para um teste.
        self.detection_time = 0
        ## Tempo de execução para n testes.
        self.mean_detection_time = 0
        ## Modelo pré-treinado.
        self.onnx_model = onnx.load(onnx_path)
        ## Carrega o preditor do modelo.
        self.predictor = prepare(self.onnx_model)
        ## Cria uma seção de inferência.
        self.ort_session = ort.InferenceSession(onnx_path)
        ## Input name.
        self.input_name = self.ort_session.get_inputs()[0].name

    ## @brief Função auxiliar que calcula a área de um vetor de bounging boxes e retorna um vetor com a área de cada uma.
    #  @param left_top Vetor de canto superio esquerdo das boxes.
    #  @param right_botton Vetor de canto inferior direito das boxes.
    #  @return Retorna um vetor com a área de cada box.
    def compute_area_of_array(self,left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    ## @brief Função auxiliar que exclui boxes muito parecidas e seleciona dentre elas a que possui a maior confiança
    #  @param boxes Vetor de boxes, cada box é um vetor com 4 elementos (x1,y1,x2,y2).
    #  @param confidences Vetor de confiança para cada box.
    #  @return Retorna um vetor com a filtragem das boxes.
    def exclude_similar_boxes(self,boxes, confidences):
        # Boxes que serão escolhidas.
        chosen_boxes = []

        # Percorre o vetor de boxes enquanto ele não estiver vazio.
        while len(boxes)>0:
            # Guarda o index da boz com maior confiaça
            max_confidence_index = np.argmax(confidences)
            # Guarda a box com maior confiaça.
            box = boxes[max_confidence_index]
            # Coloca essa box no vetor de escolhidas.
            chosen_boxes.append(box)
            # Adiciona uma dimenção para poder usar as próximas funções.
            box = np.expand_dims(box, axis=0)
            # Encontra as coordenadas da interseção entre a box escolhida e todas as outras boxes.
            overlap_left_top = np.maximum(boxes[..., :2], box[..., :2])
            overlap_right_bottom = np.minimum(boxes[..., 2:], box[..., 2:])
            overlap_area = self.compute_area_of_array(overlap_left_top, overlap_right_bottom)
            area1 = self.compute_area_of_array(boxes[..., :2], boxes[..., 2:])
            area2 = self.compute_area_of_array(box[..., :2], box[..., 2:])
            # Calcula o índice jaccard entre a box escolhida e todas as outras boxes da lista.
            jaccard = overlap_area/(area1 + area2 - overlap_area + 0.0000001)
            # Seleciona apenas as boxes que possuem índice jaccard menor que 0.5, logo possuem baixa interseção.
            mask = jaccard < 0.5
            # Retira da lista de boxes as que tiveram muita semelhança.
            confidences=confidences[mask]
            boxes=boxes[mask]

        return chosen_boxes

    ##  @brief Função que realiza o teste do algoritmo para um vetor de imagens.
    #   @param images Vetor de imagens.
    #   @param show_results Booleano que informa se deve-se mostrar ou não as imagens com bounding boxes.
    def test(self, images, show_results = False):
        # Inicializa o tempo de detecção com zero.
        self.detection_time=0

        for image in images:
            # Guarda momento que começa o processamento da imagem
            t1=time.time()

            # Armazena a altura e largura da imagem.
            h, w, _ = image.shape
            # Faz transformação de BGR para RGB.
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Faz normalização e coloca os valores de pixels entre -1 e 1.
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            # Coloca imagem no formato exigido pelo modelo.
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            # Aplica o modelo que resulta em uma lista de confiaças e boxes.
            confidences, boxes = self.ort_session.run(None, {self.input_name: img})  

            # Reduz dimensionalidade.     
            boxes = boxes[0]
            confidences = confidences[0]

            # O vetor de confianças carrega a probabilidade de ser background e a probabilidade de ser face, analisamos apenas a probabilidade de ser face,
            confidences = confidences[:, 1]
            # Esclui boxes com probabilidade de ser face menor que 70%
            mask = confidences > 0.7
            confidences  = confidences[mask]
            boxes = boxes[mask, :]
            # Reescala o tamanho das bounding boxes normalizadas para número de pixels.
            boxes[:, 0] *= w
            boxes[:, 1] *= h
            boxes[:, 2] *= w
            boxes[:, 3] *= h
            boxes = boxes.astype(np.int32)
            # Exclui boxes que são similares.
            boxes = self.exclude_similar_boxes(boxes, confidences)
            # Calcula o tempo que levou para processar uma imagem e adiciona ao tempo de detecação.
            self.detection_time += time.time()-t1

            if show_results:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.imshow("Image", image)
                cv2.waitKey(0)
        
    ##  @brief Função que avalia o desempenho de velocidade do algoritmo fazendo uma média de vários testes.
    #   @param images Vetor de imagens.
    #   @param n_tests Número de testes para calcular a média.
    def evaluate(self, images, n_tests):
        # Zera o valor médio de tempo de execução.
        self.mean_detection_time = 0
        for i in range(n_tests):
            self.test(images)
            # Adiciona o tempo de execução a cada iteração.
            self.mean_detection_time += self.detection_time
        # Calcula a média de tempo dividindo o somatório pelo número de testes.
        self.mean_detection_time = self.mean_detection_time/n_tests
        return self.mean_detection_time

#   Script para testar o código de avaliação do modelo Ultra Light Fast Generic Face Detector
if __name__ == '__main__':

    ##  @brief Função que lê todas as imagens de uma pasta e guarda em um vetor.
    def load_images_from_folder(folder):
        images = []
        filenames = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
        
        images = np.array(images)
        filenames = np.array(filenames)
        return images, filenames

    # Carrega as imagens da pasta "dataset".
    images, filenames = load_images_from_folder("dataset")
    # Cria uma instância da classe TestUltraLigtFaceDetector passando o modelo como parâmetro.
    td = TestUltraLightFaceDetector('models/onnx/version-RFB-640.onnx')
    # Avalia o desempenho da classe com 20 execuções.
    td.evaluate(images, 20)
    print("Tempo de deteccao  = {}".format(td.mean_detection_time/len(images)))

    # Mostra os resultados da detecção de todas as imagens.
    # td.test(images,True)