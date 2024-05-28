# Use the Bentoml model server with Python 3.10
FROM bentoml/model-server:0.11.0-py310

MAINTAINER ersilia

# Install dependencies
RUN pip install rdkit==2022.3.3
RUN pip install scikit-learn==0.24.2
RUN pip install scipy==1.7.1
RUN pip install numpy==1.21.2
RUN pip install pandas==1.3.3
RUN pip install cloudpickle==2.0.0
RUN pip install joblib==1.1.0
RUN pip install tqdm==4.62.2
RUN pip install matplotlib==3.4.3

# Set the working directory
WORKDIR /repo

# Copy the contents of the current directory to /repo in the Docker image
COPY . /repo

# Command to run your application (if needed)
# CMD ["python", "main.py"]