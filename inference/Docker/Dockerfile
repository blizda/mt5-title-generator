FROM pytorch/pytorch
RUN apt update && apt install -y git
RUN pip install transformers==4.2.1 onnxruntime==1.6.0 onnxruntime-tools==1.5.2 psutil wandb falcon gunicorn && pip install transformers[sentencepiece]