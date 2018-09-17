FROM jupyter/base-notebook

USER root
RUN apt-get update && apt-get install -y git wget curl
RUN apt-get install -y vim

USER $NB_UID

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

ADD irl /home/$NB_USER/inverse_rl/irl
ADD utils /home/$NB_USER/inverse_rl/utils
fix-permissions /home/$NB_USER

RUN cd /home/$NB_USER && git clone https://github.com/yrevar/simple_rl &&\
	cd simple_rl && python setup.py install

