from fastapi import UploadFile,File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from ..models.rn_model import predict


async def get_prediction(image:UploadFile = File(...)):
  print(image)
  img_matrix = await get_image(image)
  print(img_matrix)

  prediction = await predict(img_matrix)
  json_compatible_item_data = jsonable_encoder({"ok":True,"result":prediction})
  return JSONResponse(content=json_compatible_item_data)


#utils
async def get_image(image:UploadFile = File(...)):
  contents = await image.read()
  nparr = np.fromstring(contents, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return img