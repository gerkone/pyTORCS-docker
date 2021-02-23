FROM tensorflow/tensorflow:2.2.2-gpu-py3

USER root

RUN apt-get update && apt-get install -y --no-install-recommends python3


# install requirements
RUN yes | pip3 install numpy Keras gym matplotlib

RUN mkdir driver

WORKDIR driver

CMD ["/bin/bash"]
