FROM python:3.6
MAINTAINER ryoasu

RUN apt-get update && apt-get install -y git

RUN pip3 install \
    tensorflow==1.5.0 \
    keras==2.1.3 \
    pillow \
    tqdm \
    opencv-python \
    h5py

RUN git clone -b master https://github.com/ryoasu/grad-cam.git

WORKDIR /grad-cam
ENTRYPOINT ["python3", "grad_cam.py"]
CMD ["-h"]
