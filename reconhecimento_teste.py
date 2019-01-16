import os
import glob
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("recursos/indices_sala.pickle")
descritoresFaciais = np.load("recursos/descritores_sala.npy")
limiar = 0.5


# a biblioteca glob serve para percorrer uma pasta
for arquivo in glob.glob(os.path.join("fotos/teste", "*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem, 3)
    for face in facesDetectadas:
        e,t,d,b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        pontosFacias = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFacias)
        listaDescritorFacil = [fd for fd in descritorFacial]
        npArrayDescritorFacial = np.asarray(listaDescritorFacil, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        #o np.linalg calcula a distância euclidiana
        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        print("Distâncias: {}".format(distancias))
        minimo = np.argmin(distancias)
        print(minimo)
        distanciaMinima = distancias[minimo]
        print(distanciaMinima)

        if distanciaMinima <= limiar:
            nome = os.path.split(indices[minimo])[1].split(".")[0]
        else:
            nome = " "

        cv2.rectangle(imagem, (e, t), (d,b), (0,255,255),2)
        texto = "{} {:.4f}".format(nome, distanciaMinima)
        cv2.putText(imagem, texto, (d,t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,255))


    cv2.imshow("Treinamento", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()