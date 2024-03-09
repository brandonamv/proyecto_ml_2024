from st_pages import Page, show_pages
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Proyecto", layout="wide")

def manage_state():
    # Esta sección de código por ahora no hace nada
    # Si quisieran compartir variables entre views deben utilizar el session_state
    # Documentación: https://docs.streamlit.io/library/api-reference/session-state
    
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = 0
    else:
        if st.session_state["active_page"] == 1:
            # Active page was <0>

            # Delete states
            
            #del st.session_state["clinical_note"]
            pass

        st.session_state["active_page"] = 0

def ui():
    # el tercer valor es el icon que puedes usar en el sidebar de la aplicación
    # Si quieres usar otro la lista está disponible en 
    # https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json
    # Puedes sea copiar y pegar el icon como colocar el nombre entre :.
    # Ejemplo: :clipboard:
    show_pages(
        [
            Page("src/pages/01_Guess.py", "Adivina la Imagen", ":pencil:"),
            Page("src/pages/02_Captcha.py", "Captcha", ":checkered_flag:"),
            Page("src/pages/03_Cargar_modelos.py", "Cargar Modelo", ":information_source:")
        ]
    )

if __name__ == "__main__":
    manage_state()
    ui()