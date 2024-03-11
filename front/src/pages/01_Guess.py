from utils.http_service import get_predict
import streamlit as st
from PIL import Image
import os
import cv2
import random
im = Image.open("src/favicon.ico")
#globals
MODEL = None
MODEL_EXIST = False
classes = ('avion', 'perro', 'ave', 'gato',
           'ciervo', 'perro', 'sapo', 'caballo', 'baro', 'camion') 

st.set_page_config(
    "Are you a robot",
    im,
    initial_sidebar_state="expanded",
    layout="wide",
)

if "number" not in st.session_state:
    st.session_state["number"] = 0

#TODO test
def get_prediction(img):

    response = get_predict(img)
    return response 
if 'key' not in st.session_state:
    st.session_state['key'] = -1 
def image_picker():

    
    if st.button("No soy un robot"):
        st.session_state['key']=random.randint(0,9)
    if st.session_state['key']!=-1:
        uploaded_image = st.file_uploader("Cargar imagen de "+classes[st.session_state['key']], type=['png', 'jpg'])

        
        if uploaded_image is not None:
            
            file_path = save_uploaded_file(uploaded_image)
            st.write(f"Imagen guardada en: {file_path}")
            #Result
            response_predict = get_prediction(open(file_path,'rb'))
            st.write(response_predict)
            if response_predict!=classes[st.session_state['key']]:
                st.write("Usted es un robot")
            else:
                st.write("Usted es un humano")



# Función para guardar el archivo en una carpeta específica
def save_uploaded_file(uploaded_file, folder_name='img'):
    # Crear la carpeta si no existe
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Guardar el archivo en la carpeta
    file_path = os.path.join(folder_name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def main():
    image_picker()

if __name__ == "__main__":
    main()
