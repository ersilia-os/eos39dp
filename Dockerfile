FROM bentoml/model-server:0.11.0-py310

MAINTAINER ersilia

RUN pip install rdkit==2023.9.6
RUN pip install scikit-learn==1.0.2
RUN pip install scipy==1.10
RUN pip install numpy==1.26.4
RUN pip install pandas==1.4.2
RUN pip install matplotlib==3.7.5
RUN pip install tqdm==4.66.4
RUN pip install tensorflow==2.14.0

WORKDIR /repo
COPY . /repo
