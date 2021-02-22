# pyTORCS + Docker
OpenAI Gym-like, TORCS-based environment for simple autonomous driving simulations.

More info on the environment and its usage can be found on [here](https://github.com/gerkone/torcs_test/tree/master/driver).

Both the **environment** and the **agent** are designed to be run inside Docker container(s). This was done to simplify the installation, as VTORCS/TORCS can be tricky to install on some systems.

Alternatively, as the TensorFlow Docker image is quite large, the agent can be also run directly on the host, while the TORCS is still in a container.

## Installation
This project is designed to run on a Linux system, ideally with an Nvidia GPU.

The Docker image allows for easier porting on multiple systems.

### TORCS
1 **install docker** -> [guide](https://docs.docker.com/engine/install/)

Verify docker works
```
sudo docker run hello-world
```

2 **docker postinstall** -> [guide](https://docs.docker.com/engine/install/linux-postinstall/)

Additionally you can set up docker to run without sudo
```
sudo groupadd docker
sudo usermod -aG docker $USER
```
Log out and log back in so that your group membership is re-evaluated.

Eventually configure docker to start on boot.
```
sudo systemctl enable docker
```

3 **install nvidia-docker**

The NVIDIA Container Toolkit allows users to build and run GPU accelerated Docker containers.
Nvidia-docker essentially exposes the GPU to the containers to use: https://github.com/NVIDIA/nvidia-docker

To install the toolkit follow this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

### Dockerized agent
**install docker-compose** -> [guide](https://docs.docker.com/compose/install/#install-compose)

Verify docker-compose works
```
docker-compose --version
```

## Usage



## References
The python client was initially based on [ugo-nama-kun's gym_torcs](https://github.com/ugo-nama-kun/gym_torcs), with some changes and quality of life improvements.

The modified TORCS source is taken from [Giuseppe Cuccu's vtorcs](https://github.com/giuse/vtorcs), which enables 64x64 RGB vision.
