# Use a lightweight Python base image, e.g. python:3.9-slim
FROM python:3.9-slim

# 1) Install system dev libraries needed by e.g. Pillow or others
RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    # ...any other libs if needed...
    && rm -rf /var/lib/apt/lists/*

# 2) (Optional) create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3) Copy your requirements.txt
COPY requirements.txt /tmp/

# 4) Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# 5) Copy the rest of your app into the image
COPY . /app
WORKDIR /app

# 6) Set the port to the dynamic $PORT from Railway
EXPOSE 8000

# 7) Command: run Streamlit on $PORT. 
#    Railway provides the port as an env var, so we read $PORT 
CMD ["sh", "-c", "streamlit run xbp_v5.py --server.port=$PORT --server.address=0.0.0.0"]
