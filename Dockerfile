# Use a base image with FEniCS installed
FROM quay.io/fenicsproject/stable:current AS base

# Set the working directory
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install Miniconda and create environment
FROM continuumio/miniconda3 AS miniconda
COPY environment.yaml /tmp/environment.yaml
RUN conda env create --file=/tmp/environment.yaml
RUN echo "source activate navier_stokes" > ~/.bashrc

# Use the base and miniconda images
FROM base
COPY --from=miniconda /opt/conda/envs/navier_stokes /opt/conda/envs/navier_stokes

# Set the working directory
WORKDIR /app

# Define the command to run your project (replace with your actual command)
CMD ["echo", "container created"]
