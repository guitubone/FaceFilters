# Relatório Final

# Integrantes:
Guilherme Hideo Tubone - 9019403  
Lucas de Oliveira Pacheco - 9293182 

# FaceFilters:
- Detecção de faces: Encontrar com OpenCV todas as faces humanas
na foto, extraindo os retângulos que as delimitam.  
- Detecção de pontos de referência: A partir da parte da imagem
referente ao rosto, detectar olhos, boca, nariz, etc. Utilizando a biblioteca Dlib.  
- Sobreposição do filtro: Dado um filtro e os pontos de referência no
rosto, sobrepor o filtro nas pontos adequados.  

# Descrição
Em algumas redes sociais como Instagram/Snapchat existe um recurso chamado Stories, em que o usuário pode compartilhar fotos ou vídeos curtos durante 24h. Um dos recursos que existem neste tipo de aplicativo são os filtros animados, que localizam as faces das pessoas na foto e aplica efeitos visuais sobre ela. Objetivo do projeto: Fazer uma réplica deste recurso utilizando Python. O programa irá receber fotos com pessoas e então aplicar filtros sobre o rosto das pessoas.

Imagens utilizadas em /images

![Neymar](images/neymar.jpg)
![Neymar Debug](output/neymar_debug.png)  
![Neymar Blur](output/neymar_blur.png)  
![Neymar PixelSunglasses](output/neymar_pixel_sunglasses.png)  

![Selecao](images/selecao.jpg)
![Selecao GlassesAndMustache](output/selecao_glassesandmustache.png)  

![Portugal](images/portugal.jpeg)
![Portugal Dog](output/portugal_dog.png)  
