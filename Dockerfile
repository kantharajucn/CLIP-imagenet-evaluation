FROM bethgelab/deeplearning:cuda10.0-cudnn7
WORKDIR /clip_inference
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
