# pyTORCS + Docker
OpenAI Gym-like, TORCS-based environment for simple autonomous driving simulations.

The **environment** is designed to be run inside a Docker container(s). This was done to simplify the installation, as torcs/TORCS can be tricky to install on some systems.

Either way, everything can be still installed and run directly on the host.

More info on the environment and its usages can be found on [here](https://github.com/gerkone/pyTORCS-docker/tree/master/driver/torcs_client).

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

3 **install nvidia-docker** -> [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

The NVIDIA Container Toolkit allows users to build and run GPU accelerated Docker containers.
Nvidia-docker essentially exposes the GPU to the containers to use: https://github.com/NVIDIA/nvidia-docker

4 **install docker-compose** -> [guide](https://docs.docker.com/compose/install/#install-compose)

Verify docker-compose works
```
docker-compose --version
```

5a **pull the torcs image**
```
docker pull gerkone/torcs
```

5b **build the torcs image yourself**
```
docker build -t <your image name> torcs/
```

6 (optional) **install konsole**

Konsole is already shipped with every KDE installations.

On Ubuntu
```
sudo apt-get install konsole
```
On Arch/Manjaro
```
sudo pacman -S konsole
```

## Host version installation
To install TORCS follow the [guide](https://github.com/gerkone/pyTORCS-docker/blob/master/torcs/README.md) in  the torcs readme.

For the python requirements for the example simply run
```
pip install -r requirements.txt
```

## Usage
To run the example you can use the script _pytorcs.py_.
```
python pytorcs.py
```
This will start the TORCS container, open a new window with the game and start running the agent.

You can change some settings and options by editing the [simulation.yaml](config/simulation.yaml) file. For more details on the parameters and on how to use your code check [this](https://github.com/gerkone/pyTORCS-docker/blob/master/driver/torcs_client/README.md).

If you don't want to install konsole you can run it with your shell of choice with
```
python pytorcs.py --console <terminator|xterm|gnome-terminal...>
```

If you want to run the TORCS container manually you can use
```
nvidia-docker run -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY -p 3001:3001/udp -it --rm gerkone/torcs
```

## Troubleshooting and known issues
If you get the error "_freeglut (/usr/local/lib/torcs/torcs-bin): failed to open display ':0'_" OR the torcs window does not pop up after running you might need to allow access to your X display server by using
```
xhost local:root
```

## References
This torcs is a modified version of 1.3.7 with the following changes:
- The **main menu is completely skipped** and the race can be configured by using an _.xml_ file. This was done to allow a faster restart and most importantly to avoid using xautomation.
- The **countdown at the beginning of each race was removed**, to save 3 seconds each time.
- The **loading screens were also removed**. I found that this somehow saves a lot of time.
- The vision works with shared memory out-of-the-box, but I made some changes to keep it simple and readable with pure python.

The Python environment was initially based on [ugo-nama-kun's gym_torcs](https://github.com/ugo-nama-kun/gym_torcs), with some changes to accomodate docker and quality of life improvements.

The Python torcs client is an extended version of snakeoil3.
