FROM python:3.6
MAINTAINER ryoasu

RUN apt-get update && apt-get install -y git

RUN pip3 install \
    tensorflow==1.4.0 \
    keras==2.0.9 \
    pillow \
    tqdm \
    opencv-python \
    h5py

RUN git clone -b master https://github.com/ryoasu/grad-cam.git

WORKDIR /grad-cam
ENTRYPOINT ["python3", "grad_cam.py"]
CMD ["-h"]
