import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
import os

def image_picker():
    images = [
        "captcha_images/rat.png", 
        "captcha_images/bike.png", 
        "captcha_images/car.png", 
        "captcha_images/cat.png", 
        "captcha_images/chair.png",
        "captcha_images/tiger.png",
        "captcha_images/airplane.png",
        "captcha_images/monkey.png",
        "captcha_images/bear.png",
        "captcha_images/pencil.png",
        "captcha_images/panda.png",
        "captcha_images/cellphone.png"]
    img = image_select("Selecciona una imagen", images, key="clicked_images", use_container_width= False)
    selected_image = img

def main():
    image_picker()

if __name__ == "__main__":
    main()