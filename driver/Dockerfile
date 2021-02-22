FROM tensorflow/tensorflow:2.2.2-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends python3


# install requirements
RUN pip3 install numpy Keras gym
# docker python
RUN curl -fsSL "https://download.docker.com/linux/ubuntu/gpg" | apt-key add -
RUN add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_r\elease -cs) \
    stable"
RUN apt update && apt install -y --no-install-recommends docker-ce
RUN pip3 install docker


ENTRYPOINT ["sh", "setup.sh"]
