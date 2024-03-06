import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
import os

im = Image.open("src/favicon.ico")

st.set_page_config(
    "EsKape Room",
    im,
    initial_sidebar_state="expanded",
    layout="wide",
)

if "number" not in st.session_state:
    st.session_state["number"] = 0

def image_picker():
    images = ["src/img/perro.png", "src/img/lobo.png", "src/img/caballo.png"]
    img = image_select("Selecciona una imagen", images, key="clicked_images", use_container_width= False)
    selected_image = img
    
    uploaded_image = st.file_uploader("Cargar imagen", type=['png', 'jpg'])
    # Guardar los archivos cargados
    selected_uploaded_image = ''
    if uploaded_image is not None:
        file_path = save_uploaded_file(uploaded_image)
        st.write(f"Imagen guardada en: {file_path}")
        selected_uploaded_image = file_path

    image_options(selected_image, selected_uploaded_image)

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

def image_options(image_from_selector, image_uploaded):
    option = st.sidebar.radio("¿Cual imagen quieres usar?", ("Imagen del selector", "Imagen Cargada"))

    if option == "Imagen del selector":
        st.image(image_from_selector)

    if option == "Imagen Cargada" and image_uploaded != '':
        st.image(image_uploaded)

def main():
    image_picker()

if __name__ == "__main__":
    main()
