import os
import streamlit as st
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model= Net()
model = torch.load('/src/utils/model.joblib')
model.eval()

im = Image.open("src/favicon.ico")
st.set_page_config(
    "EsKape Room",
    im,
    initial_sidebar_state="expanded",
    layout="wide",
)

# Función para guardar el archivo en una carpeta específica
def save_uploaded_file(uploaded_file, folder_name='src/models/output'):
    # Crear la carpeta si no existe
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Guardar el archivo en la carpeta
    file_path = os.path.join(folder_name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def cargar_modelos_ui():
    # Título de la aplicación
    st.title('Cargador de Modelos de Machine Learning')

    # Cargadores de archivos
    uploaded_file1 = st.file_uploader("Cargar archivo 1 (.joblib or .pickle)", type=['joblib', 'pickle'])
    uploaded_file2 = st.file_uploader("Cargar archivo 2 (.joblib or .pickle)", type=['joblib', 'pickle'])

    # Guardar los archivos cargados
    if uploaded_file1 is not None:
        file_path = save_uploaded_file(uploaded_file1)
        st.write(f"Modelo guardado en: {file_path}")

    if uploaded_file2 is not None:
        file_path = save_uploaded_file(uploaded_file2)
        st.write(f"Modelo guardado en: {file_path}")


def main():
    cargar_modelos_ui()

if __name__ == "__main__":

    main()