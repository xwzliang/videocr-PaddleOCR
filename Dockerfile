# FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda11.8-cudnn8.9

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ffmpeg \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /workspace

# # Install build tools
# RUN pip install --upgrade pip setuptools wheel

# # RUN pip install paddlepaddle-gpu

# # Copy project metadata first
# COPY pyproject.toml .
# COPY . .

# # Install the package (including dependencies)
# RUN pip install .

# CMD ["python"]

FROM xwzliang/videocr_paddle:0.0.2
RUN pip install av