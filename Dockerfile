FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

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
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Copy environment file
COPY environment.yml /app/environment.yml

# Create the Conda environment
RUN conda env create -f /app/environment.yml

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Expose the port
EXPOSE 5000

# Set the default command to run the application
CMD ["conda", "run", "--no-capture-output", "-n", "customer-service-project", "python", "main.py"]
