# pyTORCS + Docker
OpenAI Gym-like, TORCS-based environment for simple autonomous driving simulations.

Both the **environment** and the **agent** are designed to be run inside a Docker container(s). This was done to simplify the installation, as VTORCS/TORCS can be tricky to install on some systems.

Either can be still installed and run directly on the host. 

More info on the environment and its usages can be found on [here](https://github.com/gerkone/torcs_test/tree/master/driver).

## Dockerized version installation
This project is designed to run on a Linux system, ideally with an Nvidia GPU.

The Docker image allows for easier porting on multiple systems.

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

4 **install docker-compose** -> [guide](https://docs.docker.com/compose/install/#install-compose)

Verify docker-compose works
```
docker-compose --version
```

## Host version installation
To install TORCS follow the [guide](https://github.com/gerkone/pyTORCS-docker/blob/master/vtorcs/README.md) in  the vtorcs readme.

For the python requirements for the example simply run
```
pip install -r requirements.txt
```
in the _driver/_ folder

## Usage
To run the example you can use the script _torcs_test.py_.
```
python torcs_test.py
```
This will start two containers, one for TORCS and one for the agent, and connect a terminal to the agent output to monitor training.

To run the agent on the host the flag _-d_ needs to be set

```
python torcs_test.py -d
```

To run TORCS on the host the flag _-t_ needs to be set

```
python torcs_test.py -t
```

If you want to run the TORCS container manually you can use
```
docker run -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY -p 3101:3101/udp -it getkone/torcs
```

## References
The python client was initially based on [ugo-nama-kun's gym_torcs](https://github.com/ugo-nama-kun/gym_torcs), with some changes yo accomodate docker and quality of life improvements.

The modified TORCS source is taken from [Giuseppe Cuccu's vtorcs](https://github.com/giuse/vtorcs), which enables 64x64 RGB vision.
