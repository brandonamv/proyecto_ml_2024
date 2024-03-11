import io
import os
import cv2
import numpy as np
import requests
import json
url = 'http://api:8080/predict'


def get_predict(img):
  # Check if img is not empty (optional)
  x = requests.post(url=url, files={'image':img}) 
  return x.text.split('"')[5]