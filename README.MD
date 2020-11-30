### Reconhecimento Facial com PCA Eigen Faces.

Solicitado pelo Professor Márcio Koch como método de avaliação da disciplina de Aprendizado de Máquina Não Supervisionado, este trabalho utiliza o OpenCV para reconhecimento em um dataset com imagens de rosto.
A base de treinamento é de 70% do dataset. O dataset também conta com 10 imagens pessoais ao final.

Para executar, é necessário instalação do OpenCV:
   ``` 
   python opencv-contrib-python"
   ```
Utilizar Python 3.7 ou superior.

Após instalação do OpenCV e Python > 3.7, executar o arquivo: 
  ```
 python PCA.py
  ```
 
Saída: 

_________________________
PCA:  10  componentes
Taxa de acerto:  91.869918699187 %
_________________________
PCA:  11  componentes
Taxa de acerto:  94.3089430894309 %
_________________________
PCA:  12  componentes
Taxa de acerto:  95.9349593495935 %
_________________________
PCA:  13  componentes
Taxa de acerto:  94.3089430894309 %
_________________________
PCA:  14  componentes
Taxa de acerto:  94.3089430894309 %
_________________________
PCA:  15  componentes
Taxa de acerto:  95.1219512195122 %
_________________________
PCA:  16  componentes
Taxa de acerto:  95.1219512195122 %
_________________________
PCA:  17  componentes
Taxa de acerto:  95.1219512195122 %
_________________________
PCA:  18  componentes
Taxa de acerto:  95.1219512195122 %
_________________________
PCA:  19  componentes
Taxa de acerto:  95.1219512195122 %
_________________________
PCA:  20  componentes
Taxa de acerto:  94.3089430894309 %
Treino:  287  imagens
Teste:  123  imagens