FROM jupyter/base-notebook

USER root
RUN apt-get update && apt-get install -y git wget curl
RUN apt-get install -y vim
# ADD --chown=0:0 https://raw.githubusercontent.com/jupyter/docker-stacks/master/base-notebook/fix-permissions /usr/local/bin/fix-permissions
# RUN chmod +rx /usr/local/bin/fix-permissions

USER $NB_UID

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

# ADD irl /home/$NB_USER/inverse_rl/irl
# ADD utils /home/$NB_USER/inverse_rl/utils
# RUN fix-permissions /home/$NB_USER

WORKDIR /home/$NB_USER
RUN git clone https://github.com/yrevar/simple_rl
WORKDIR /home/$NB_USER/simple_rl
RUN python setup.py install

WORKDIR /home/$NB_USER
RUN git clone https://github.com/yrevar/inverse_rl.git
