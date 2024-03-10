FROM python:3.10-buster
EXPOSE 8080

RUN apt-get update
RUN apt-get install -y libsndfile1-dev
RUN apt-get install -y ffmpeg

RUN pip install fastapi
RUN pip install "uvicorn[standard]"
RUN pip install poetry
RUN pip install python-multipart
RUN pip install numpy
RUN pip install opencv-python
RUN pip install joblib
RUN pip install torch 
RUN pip install torchvision

WORKDIR /usr/app
COPY . .
#RUN pip install -r requirements-hw.txt
CMD ["poetry", "run","uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]