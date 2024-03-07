import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
import os

def image_picker():
    images = [
        "src/captcha_images/rat.png", 
        "src/captcha_images/bike.png", 
        "src/captcha_images/car.png", 
        "src/captcha_images/cat.png", 
        "src/captcha_images/chair.png",
        "src/captcha_images/tiger.png",
        "src/captcha_images/airplane.png",
        "src/captcha_images/monkey.png",
        "src/captcha_images/bear.png",
        "src/captcha_images/pencil.png",
        "src/captcha_images/panda.png",
        "src/captcha_images/cellphone.png"]
    img = image_select("Selecciona una imagen", images, key="clicked_images", use_container_width= False)
    selected_image = img
def main():
    image_picker()

if __name__ == "__main__":
    main()