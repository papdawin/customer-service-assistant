# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
ENV CI=true
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
ENV CONDA_ALWAYS_YES=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    build-essential \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Copy environment file
COPY environment.yml /app/environment.yml

# Create the Conda environment
RUN conda install --name base conda-anaconda-tos && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f /app/environment.yml

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Copy the model into the container
COPY /app/models /app/models

# Expose the app port
EXPOSE 5000

# Set the default command to run the application
CMD ["conda", "run", "--no-capture-output", "-n", "customer-service-project", "python", "main.py"]
