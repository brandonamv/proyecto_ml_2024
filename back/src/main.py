from typing import Union

from fastapi import FastAPI,UploadFile, File
from .service.models import get_prediction
app = FastAPI()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    return await get_prediction(image)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

