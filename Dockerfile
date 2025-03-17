FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Налаштування таймзони
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Bishkek

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libfontconfig1 \
    libxrender1 \
    libxtst6 \
    libgl1-mesa-glx \
    libturbojpeg \
    python3-setuptools \
    python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /project

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python3", "app/server.py"]