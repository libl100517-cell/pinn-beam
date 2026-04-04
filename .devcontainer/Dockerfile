FROM python:3.11

RUN apt-get update && apt-get install -y \
    git curl wget vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    numpy matplotlib scipy pytest

# Make ALL python paths point to python3.11 (which has packages)
# Debian base image ships python3.13 at /usr/bin/python3.13 which VS Code may pick
RUN ln -sf /usr/local/bin/python3 /usr/bin/python3 && \
    ln -sf /usr/local/bin/python  /usr/bin/python  && \
    ln -sf /usr/local/bin/python3.11 /usr/bin/python3.13
