# FaceFilters

# Integrantes:
Guilherme Hideo Tubone - 9019403  
Lucas de Oliveira Pacheco - 9293182 

# Dependências
OpenCV  
Dlib  
Baixar e copiar para diretório /data: https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat  

# Resumo
Área do projeto: Aprendizado de características  
Contextualização do problema: Em algumas redes sociais como
Instagram/Snapchat existe um recurso chamado Stories, em que o usuário
pode compartilhar fotos ou vídeos curtos durante 24h. Um dos recursos que
existem neste tipo de aplicativo são os filtros animados, que localizam as
faces das pessoas na foto e aplica efeitos visuais sobre ela.
Objetivo do projeto: Fazer uma réplica deste recurso utilizando Python. O
programa irá receber fotos com pessoas e então aplicar filtros sobre o rosto
das pessoas.  
Etapas:  
- Detecção de faces: Encontrar com OpenCV todas as faces humanas
na foto, extraindo os retângulos que as delimitam.  
- Detecção de pontos de referência: A partir da parte da imagem
referente ao rosto, detectar olhos, boca, nariz, etc. Utilizando a biblioteca Dlib.  
- Sobreposição do filtro: Dado um filtro e os pontos de referência no
rosto, sobrepor o filtro nas pontos adequados.  
