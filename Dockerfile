# Use the Bentoml model server with Python 3.10
FROM bentoml/model-server:0.11.0-py38

MAINTAINER ersilia

# Install dependencies
RUN pip install rdkit==2023.9.6
RUN pip install scikit-learn==0.24.2
RUN pip install scipy==1.10
RUN pip install cloudpickle==3.0.0
RUN pip install numpy==1.24.4
RUN pip install pandas==1.3.3
RUN pip install matplotlib==3.7.5
RUN pip install tqdm==4.66.4

# Set the working directory
WORKDIR /repo

# Copy the contents of the current directory to /repo in the Docker image
COPY . /repo
