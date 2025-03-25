import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def capture_images(output_dir="dataset", num_images=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    object_name = input("Digite o nome do objeto: ").strip()
    object_path = os.path.join(output_dir, object_name)
    os.makedirs(object_path, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return
    
    print(f"Capturando {num_images} imagens para {object_name}...")
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar a imagem.")
            break
        
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            img_path = os.path.join(object_path, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Imagem {count + 1}/{num_images} salva em {img_path}")
            count += 1
        elif key == ord('q'):
            print("Captura cancelada.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Captura concluída!")

def load_data(dataset_path="dataset", img_size=(64, 64)):
    images, labels = [], []
    class_names = sorted(os.listdir(dataset_path))
    
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)
    return images, labels, class_names

def train_model(dataset_path="dataset", img_size=(64, 64)):
    images, labels, class_names = load_data(dataset_path, img_size)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    model.save("object_recognition_model.h5")
    print("Modelo treinado e salvo como object_recognition_model.h5")
    return model, class_names

def predict_object(model_path="object_recognition_model.h5", dataset_path="dataset", img_size=(64, 64)):
    model = keras.models.load_model(model_path)
    class_names = sorted(os.listdir(dataset_path))
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return
    
    print("Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar a imagem.")
            break
        
        img = cv2.resize(frame, img_size)
        img = np.expand_dims(img, axis=0) / 255.0
        prediction = model.predict(img)
        class_id = np.argmax(prediction)
        label = class_names[class_id]
        
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Object Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Reconhecimento finalizado.")

if __name__ == "__main__":
    choice = input("Digite 'c' para capturar imagens, 't' para treinar o modelo ou 'p' para prever objetos: ").strip().lower()
    if choice == 'c':
        capture_images()
    elif choice == 't':
        train_model()
    elif choice == 'p':
        predict_object()
    else:
        print("Opção inválida.")
