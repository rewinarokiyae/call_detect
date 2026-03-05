FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies. We use the PyTorch CPU index to keep the image size reasonable
# since it is being shared, unless a GPU is explicitly required.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir streamlit

# Copy the rest of the application files
COPY . .

# Prevent Python from buffering standard output and error streams
ENV PYTHONUNBUFFERED=1

# Expose standard ports (8501 for Streamlit UI, 8000 for FastAPI if used independently)
EXPOSE 8501
EXPOSE 8000

# Run the UI app as the default entrypoint
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
