# Use the Bentoml model server with Python 3.10
FROM bentoml/model-server:0.11.0-py310

MAINTAINER ersilia

# Switch to root user to install system dependencies
USER root

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user (default in Bentoml model server images)
USER bentoml

# Install compatible versions of setuptools and wheel
RUN pip install setuptools==59.5.0 wheel

# Install dependencies with pinned versions
RUN pip install rdkit==2022.3.3
RUN pip install scikit-learn==0.24.2
RUN pip install scipy==1.7.1
RUN pip install numpy==1.21.2
RUN pip install pandas==1.3.3
RUN pip install cloudpickle==2.0.0
RUN pip install joblib==1.1.0
RUN pip install tqdm==4.62.2
RUN pip install matplotlib==3.4.3
RUN pip install cython==0.29.23

# Set the working directory
WORKDIR /repo

# Copy the contents of the current directory to /repo in the Docker image
COPY . /repo

# Compile Cython files
RUN find . -name "*.pyx" -exec cythonize -i {} +

# Ensure that any .pyx files are compiled and any potential issues with C extensions are resolved
RUN python setup.py build_ext --inplace

# Command to run your application (if needed)
# CMD ["python", "main.py"]