import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image as grad_cam_preprocess_image
import torch.nn.functional as F
import streamlit as st

# Configuration de l'appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prétraitement
def preprocess_images(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  
    return image_tensor

# Définition du modèle CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 112 * 112, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 112 * 112) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Fonction pour calculer la précision
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Entraînement du modèle
def train_model(train_data, validation_data):
    model = SimpleCNN().to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):  
        model.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_accuracy = calculate_accuracy(model, train_data)
        validation_accuracy = calculate_accuracy(model, validation_data)
        print(f'Époque {epoch+1} - Précision Entraînement: {train_accuracy}%, Précision Validation: {validation_accuracy}%')

    torch.save(model.state_dict(), 'model_V2.pth')
    return model

# Chargement ou entraînement du modèle
def get_model():
    try:
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load('model_V2.pth', map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Le modèle pré-entraîné n'a pas été trouvé. Commencer l'entraînement.")
        train_data = preprocess_images('C:/Users/HP/Documents/Breast_Cancer_Project/Classification_Dataset/train')
        validation_data = preprocess_images('C:/Users/HP/Documents/Breast_Cancer_Project/Classification_Dataset/val')
        model = train_model(train_data, validation_data)
    return model

# Préparation de l'image pour Grad-CAM
def prepare_image_for_grad_cam(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    rgb_img = np.float32(img) / 255
    # Important: Nous retournons l'image RGB prétraitée pour l'affichage Grad-CAM
    return rgb_img

# Application de Grad-CAM
def apply_grad_cam(model, rgb_img):
    target_layers = [model.conv1]
    # Nous prétraitons ici l'image pour créer le tenseur d'entrée pour Grad-CAM
    input_tensor = grad_cam_preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
    # Conversion du tenseur d'entrée en tableau numpy pour l'affichage
    image_with_cam = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return image_with_cam

# Interface utilisateur Streamlit
def run_streamlit():
    st.title('Détection du Cancer du Sein à partir de Biopsies')

    model = get_model()

    uploaded_file = st.file_uploader("Choisissez une image de biopsie", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Image téléchargée', use_column_width=True)

        if st.button('Prédire'):
            with st.spinner("Modèle en cours d'exécution..."):
                img_tensor = preprocess_images(image).to(device)

                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                classes = ['Bénin', 'In Situ', 'Invasif', 'Normal']
                predicted_class = classes[predicted.item()]
                st.success(f"Prédiction : {predicted_class}")

                # Convertir l'image PIL en RGB pour Grad-CAM
                rgb_img = prepare_image_for_grad_cam(image)
                cam_image = apply_grad_cam(model, rgb_img)
                st.image(cam_image, caption='Visualisation Grad-CAM', use_column_width=True)

if __name__ == '__main__':
    run_streamlit()
