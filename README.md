# visao-computacional

# Explicação do Código de Reconhecimento de Objetos com Visão Computacional

## Visão Geral
Este código implementa um sistema de reconhecimento de objetos utilizando a biblioteca OpenCV para captura de imagens, TensorFlow/Keras para criação e treinamento de um modelo de aprendizado profundo e Scikit-learn para divisão do conjunto de dados. O fluxo do programa consiste em três etapas principais:
1. Captura de imagens via webcam.
2. Treinamento de um modelo de CNN (rede neural convolucional) para classificação das imagens.
3. Uso do modelo treinado para reconhecimento em tempo real.

## Importação de Bibliotecas
```python
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
```
O código importa bibliotecas essenciais para visão computacional (`cv2`), manipulação de arquivos (`os`), cálculos numéricos (`numpy`), aprendizado profundo (`tensorflow.keras`) e divisão de dados (`train_test_split`).

## 1. Captura de Imagens
```python
def capture_images(output_dir="dataset", num_images=20):
```
Esta função permite capturar e armazenar imagens para treinar o modelo.

### Etapas:
- **Criação de diretório**: Se não existir, cria um diretório `dataset`.
- **Solicitação do nome do objeto**: O usuário fornece um nome, que será usado para rotular as imagens.
- **Abertura da câmera**: Usa `cv2.VideoCapture(0)` para acessar a webcam.
- **Loop de captura**: Captura `num_images` imagens quando a tecla `s` for pressionada.
- **Armazenamento das imagens**: Salva as imagens numeradas dentro do diretório do objeto.
- **Finalização**: Libera a câmera e fecha a janela.

## 2. Carregamento dos Dados
```python
def load_data(dataset_path="dataset", img_size=(64, 64)):
```
Esta função carrega e processa as imagens para treinamento.

### Etapas:
- **Lista as classes**: Obtém os nomes dos diretórios dentro de `dataset` (cada classe é um objeto diferente).
- **Lê e redimensiona as imagens**: Converte cada imagem para o tamanho `64x64` pixels.
- **Normalização**: Transforma os valores dos pixels de `[0, 255]` para `[0, 1]`.
- **Retorna os dados**: Retorna as imagens e seus rótulos correspondentes.

## 3. Treinamento do Modelo
```python
def train_model(dataset_path="dataset", img_size=(64, 64)):
```
Esta função treina uma rede neural convolucional para classificar objetos.

### Etapas:
1. **Carregamento dos dados**: Obtém imagens e rótulos usando `load_data()`.
2. **Divisão dos dados**: Separa em treino (`80%`) e teste (`20%`).
3. **Criação do modelo**:
   - Camadas convolucionais (`Conv2D`) para extração de características.
   - Camadas de pooling (`MaxPooling2D`) para redução da dimensionalidade.
   - Camada densa (`Dense`) para classificação.
4. **Compilação**: Usa o otimizador `adam` e a função de perda `sparse_categorical_crossentropy`.
5. **Treinamento**: Executa `model.fit()` por 10 épocas.
6. **Salvamento do modelo**: Armazena o modelo treinado como `object_recognition_model.h5`.

## 4. Predição de Objetos em Tempo Real
```python
def predict_object(model_path="object_recognition_model.h5", dataset_path="dataset", img_size=(64, 64)):
```
Esta função utiliza o modelo treinado para identificar objetos capturados pela câmera.

### Etapas:
1. **Carrega o modelo salvo**.
2. **Obtém os nomes das classes**.
3. **Abre a câmera**.
4. **Loop de captura**:
   - Obtém um frame da câmera.
   - Redimensiona e normaliza a imagem.
   - Faz uma previsão com `model.predict()`.
   - Exibe o nome do objeto identificado na tela.
5. **Fecha a câmera ao pressionar 'q'**.

## 5. Execução do Programa
```python
if __name__ == "__main__":
```
Este bloco permite que o usuário escolha entre:
- **Capturar imagens (`c`)**.
- **Treinar o modelo (`t`)**.
- **Realizar previsões (`p`)**.

Se a opção for inválida, exibe uma mensagem de erro.

## Conclusão
Este código fornece um pipeline básico para reconhecimento de objetos com visão computacional. Ele permite capturar imagens, treinar um modelo e fazer previsões em tempo real. Melhorias podem incluir:
- Aumento do número de imagens por classe.
- Uso de redes neurais mais complexas (ex: ResNet, MobileNet).
- Implementação de técnicas de aumento de dados para melhorar a precisão.

