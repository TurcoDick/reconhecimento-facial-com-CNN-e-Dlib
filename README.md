# reconhecimento-facial-com-CNN-e-Dlib #
Sistema básico para reconhecimento facial usando openCV2 e Dlib. Primeiramente fiz uma comparação entre os algoritmos de detecção de faces Hog, Haas Cascade e CNN, onde cada algoritmo teve resultados bons dependendo da imagem.<br/><br/>

Foram utilizadas fotos aleátorias de torcidas de times de futebol para fazer a comparação entre os algoritmos Hog, Haas Cascade e CNN.<br/><br/>

também foram utilizadas algumas fotos do Neymar, Barack Obama e do Trump que usei para treinar o algoritmo.<br/>

 ### Instruções de uso <br/>
 ### Importante: Todas as fotos utilizadas devem ter a extensão .jpg <br/>
  1º- execute o comando: pip install -r requirements.txt , ele vai instalar as dependencias requiridas, foi utilizado o Python 3 para este projeto. <br/><br/>
  2º - Para executar a comparação execute o arquivo: comparativo_haar_hog_cnn.py , vai abrir uma janela com uma foto, tecle enter para ir para a próxima.<br/>
  3º - Para treinar o algoritmo, coloque algumas novas fotos no diretório treinamento, as fotos devem ter somente uma pessoa em cada e o nome deve ser o da pessoal a ser identificada, execute o arquivo: reconhecimento_facial_treinamento.py<br/> 
  4º - Para rodar o teste coloque fotos da pessoa a ser identificada no diretório teste (essas fotos não podem ser as fotos que foram utilizadas para teste), execute o arquivo: reconhecimento_teste.py 

