# FROM bentoml/model-server:0.11.0-py310

# MAINTAINER ersilia

# # Install dependencies
# RUN pip install --no-cache-dir rdkit==2022.3.3 \
#     scipy==1.7.1 \
#     numpy==1.26.4 \
#     cloudpickle==2.0.0 \
#     joblib==1.1.0 \
#     tqdm==4.66.4 \
#     matplotlib==3.4.3 \
#     cython==0.29.23

# # Set the working directory
# WORKDIR /repo

# # Copy the contents of the current directory to /repo in the Docker image
# COPY . /repo

# # Command to run your application (if needed)
# # CMD ["python", "main.py"]
# Use the specified base image
FROM bentoml/model-server:0.11.0-py310

# Maintainer information
MAINTAINER ersilia

# Install dependencies without using cache to reduce image size
RUN pip install --no-cache-dir rdkit==2022.09.5 \
    scipy==1.7.1 \
    numpy==1.26.4 \
    cloudpickle==2.0.0 \
    joblib==1.1.0 \
    tqdm==4.66.4 \
    matplotlib==3.4.3 \
    cython==0.29.23

# Set the working directory
WORKDIR /repo

# Copy the contents of the current directory to /repo in the Docker image
COPY . /repo

# Uncomment and specify the command to run your application if needed
# CMD ["python", "main.py"]
