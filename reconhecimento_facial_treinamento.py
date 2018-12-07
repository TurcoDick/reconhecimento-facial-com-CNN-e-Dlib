import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descritoresFaciais = None

# a biblioteca glob serve para percorrer uma pasta
for arquivo in glob.glob(os.path.join("fotos/treinamento", "*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem, 1)
    numeroFacesdetectadas = len(facesDetectadas)

    if numeroFacesdetectadas > 1:
        print("Há mais de uma face na imagem {}".format(arquivo))
        exit(0)
    elif numeroFacesdetectadas < 1:
        print("Nenhuma face encontrada no arquivo {}".format(arquivo))
        exit(0)

    for face in facesDetectadas:
        pontosFaciais = detectorPontos(imagem, face)

        # o descritorFacil são 128 caractéristicas que o CNN selecionou como as mais importantes de cada imagem
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)

        # mostra o nome do arquivo
        #print(format(arquivo))

        # mostra a quantidade de caracteristicas concideradas principais pela CNN
        #print(len(descritorFacial))

        # mostra a matriz de caracteristicas
        listaDescritorFacial = [df for df in descritorFacial]
        #print(listaDescritorFacial)

        # agora é necessário converter esta lista em um vetor do tipo numpy
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        #print(npArrayDescritorFacial)

        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
        #print(npArrayDescritorFacial)

        if descritoresFaciais is None:
            descritoresFaciais = npArrayDescritorFacial
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial),axis=0)

        indice[idx] = arquivo
        idx +=1


    cv2.imshow("Treinamento", imagem)
    cv2.waitKey(0)

print("Tamanho: {} Formato: {}".format(len(descritoresFaciais), descritoresFaciais.shape))

np.save("recursos/descritores_sala.npy", descritoresFaciais)
with open("recursos/indices_sala.pickle", "wb") as f:
    cPickle.dump(indice, f)

cv2.destroyAllWindows()