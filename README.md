# reconhecimento-facial-com-CNN-e-Dlib #
Sistema básico para reconhecimento facial usando openCV2 e Dlib. Primeiramente fiz uma comparação entre os algoritmos de detecção de faces Hog, Haas Cascade e CNN, onde cada algoritmo teve resultados bons dependendo da imagem. Em seguida foi implementado um algoritmo para treinar e testar o reconhecimento de faces.<br/><br/>

 É importante frizar que: Detecção de faces significa que foram encontradas características que indicam que na quela imagem contem uma ou mais faces, já o reconhecimento de faces significa que após a detecção de uma ou mais faces elas foram reconhecidas por caracteristicas obtidas no treinamento.<br/><br/> 

Foram utilizadas fotos aleátorias de torcidas de times de futebol para fazer a comparação entre os algoritmos Hog, Haas Cascade e CNN, pelo console é possível ver o tempo e a quantidade que cada algoritmo consiguiu obter, nas literaturas que li foi dito que o algoritmo CNN (Rede neural convolucional) seria bem superior ao Haas cascade e ao Hog, mas em algumas fotos com muitas pessoas isso nem sempre foi verdade nos testes que fiz, porém minha hipótese é que isso se deu por que as redes neurais precisam de um poder computacional consideravél para ter bons resultados, e como digo abaixo o meu computador é mediano.<br/><br/>

Também foram utilizadas algumas fotos do Neymar, Barack Obama e do Trump que usei para treinar o algoritmo.<br/>

 ### Instruções de uso <br/>
 #### Importante: Todas as fotos utilizadas devem ter a extensão .jpg #### <br/><br/>
  1º- execute o comando: pip install -r requirements.txt , ele vai instalar as dependencias requeridas, foi utilizado o Python 3 para este projeto. <br/><br/>
  2º - Para executar a comparação execute o arquivo: comparativo_haar_hog_cnn.py , vai abrir uma janela com uma foto, tecle enter para ir para a próxima.<br/><br/>
  3º - Para treinar o algoritmo, coloque algumas novas fotos no diretório treinamento, as fotos devem ter somente uma pessoa em cada e o nome deve ser o da pessoal a ser identificada, execute o arquivo: reconhecimento_facial_treinamento.py<br/><br/> 
  4º - Para rodar o teste coloque fotos da pessoa a ser identificada no diretório teste (essas fotos não podem ser as fotos que foram utilizadas para teste), execute o arquivo: reconhecimento_teste.py <br/><br/>
  
  ## Características do Computador utilizado ##<br/>
  Notebook 8 GB RAM;<br/>
  processador: i7, 4ª Geração; <br/>
  placa de video: geforce M920; <br/>
  SSD
  sistema operacional: Ubuntu 18.04 <br/>
  
  
  

