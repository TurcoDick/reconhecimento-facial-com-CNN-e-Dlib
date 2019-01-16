import cv2
import dlib
import glob
import os
import time

fonte = cv2.FONT_HERSHEY_COMPLEX
print("sssssssssssssssssssss")
for arquivo in glob.glob(os.path.join("fotos/comparacao", "*.jpg")):

    imagem = cv2.imread(arquivo)

    # Haar
    inicio_Haar = time.time()
    detectorHaar = cv2.CascadeClassifier("recursos/haarcascade_frontalface_default.xml")
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadasHaar = detectorHaar.detectMultiScale(imagemCinza, scaleFactor=1.1, minSize=(10,10))
    fim_haar = time.time()
    print("")
    print("O Haar executou em tempo de {} e detectou {} faces:".format((fim_haar - inicio_Haar), len(facesDetectadasHaar)))

    # Hog
    inicio_Hog = time.time()
    detectorHog = dlib.get_frontal_face_detector()
    facesDetectadasHog = detectorHog(imagem, 2)
    fim_hog = time.time()
    print("O HOG executou em tempo de {} e detectou {} faces:".format((fim_hog - inicio_Hog), len(facesDetectadasHog)))

    # CNN
    inicio_CNN = time.time()
    detectorCNN = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat")
    facesDetectadasCNN = detectorCNN(imagem, 1)
    fim_CNN = time.time()
    print("O CNN executou em tempo de {} e detectou {} faces:".format((fim_CNN - inicio_CNN), len(facesDetectadasCNN)))

    for (x,y,l,a) in facesDetectadasHaar:
        cv2.rectangle(imagem, (x,y), (x + l, y + a), (0,255,0),2)
        cv2.putText(imagem, "Haar", (x,y - 5), fonte, 0.5, (0,255,0))

    for face in facesDetectadasHog:
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        cv2.rectangle(imagem, (e,t), (d,b), (0,255,255),2)
        cv2.putText(imagem, "Hog", (d,t), fonte, 0.5, (0,255,255))

    for face in facesDetectadasCNN:
        e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()),
                         int(face.rect.right()), int(face.rect.bottom()),
                         face.confidence)
        cv2.rectangle(imagem, (e,t), (d,b), (255,255,0),2)
        cv2.putText(imagem, "CNN", (d, t), fonte, 0.5, (255, 255, 0))

    cv2.imshow("Comparando os detectores de imagem", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()

