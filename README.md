# pyTORCS + Docker

![PPO in lagunaseca](/media/ppo.gif)

  * [Features](#features)
  * [Installation](#installation)
  * [Host installation](#host-installation)
  * [Usage](#usage)
  * [Envoronment](#environment)
  * [Troubleshooting and known issues](#troubleshooting-and-known-issues)
  * [References](#references)

## Features
* OpenAI gym style api
* supports RGB vision and standard _scr_server_ [sensors](https://github.com/gerkone/pyTORCS-docker/tree/master/driver#sensor-description)
* torcs is run on Docker, for a simpler installation and configuration
* **no need for xautomation**, as the menu is skipped automatically
* extensive configuration though _.yaml_ files (eg car and track can be changed through a parameter). More info [here](https://github.com/gerkone/pyTORCS-docker/tree/master/driver/torcs_client#usage)
* latest torcs version (1.3.7)

## Installation
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

4a **pull the torcs image**
```
docker pull gerkone/torcs
```

4b **build the torcs image yourself**
```
docker build -t <your image name> torcs/
```

5 **install python requirements**
```
pip install -r requirements.txt
```

(optional) **install konsole**

Konsole is already shipped with every KDE installation.

On Ubuntu
```
sudo apt-get install konsole
```
On Arch/Manjaro
```
sudo pacman -S konsole
```

## Host installation
It is possible to install TORCS on the host without using docker. To do so follow this guide [guide](https://github.com/gerkone/pyTORCS-docker/blob/master/torcs/README.md). Pay attention to the configuration section.

## Usage
To run the example you can use the script _pytorcs.py_.
```
python pytorcs.py
```
This will start the TORCS container, open a new window with the game and start running the agent.

You can change some settings and options by editing the [simulation.yaml](config/simulation.yaml) file.

For more details on how the action and state space work check [this](https://github.com/gerkone/pyTORCS-docker/blob/master/driver/README.md).

For more details on the parameters and on how to use your test code and custom algorithm check [this](https://github.com/gerkone/pyTORCS-docker/blob/master/driver/torcs_client/README.md).

You can run pytorcs in a detached shell with your emulator of choice with
```
python pytorcs.py --console <konsole|terminator|xterm|gnome-terminal...>
```

If you want to run the TORCS container manually you can use
```
nvidia-docker run --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY -p 3001:3001/udp -it --rm -d gerkone/torcs
```
## Environment
More info on the environment and its usages can be found on [here](https://github.com/gerkone/pyTORCS-docker/tree/master/driver/torcs_client).

## Troubleshooting and known issues
  - __The TORCS window does not pop up__ (OR error "_freeglut (/usr/local/lib/torcs/torcs-bin): failed to open display ':0'_").

  Allow access to your X display server by using
  ```
  xhost local:root
  ```

- __Failed to initialize NVML: Unknown Error__

  Docker can't access the GPU, usually due to permissions. Run pytorcs with ```--privileged```.

  Common when using the _nvidia-docker_ AUR package. For more info check [this](https://bbs.archlinux.org/viewtopic.php?id=266915) discussion.


## References
### TORCS
This torcs is a modified version of 1.3.7 taken from [here](https://github.com/fmirus/torcs-1.3.7).

I made the following changes to the source:
- The **main menu is completely skipped** and the race can be configured by using an _.xml_ file. This was done to allow a faster restart and most importantly to avoid using xautomation.
- The **countdown at the beginning of each race was removed**, to save 3 seconds each time.
- The vision works with shared memory out-of-the-box, but I made some changes to keep it simple and readable with pure python.
- Deleted most of the cars and tracks to lighten the build.

### scr_server
The torcs server used is _scr_server_ by Daniele Loiacono et al.

The Python-side client is an extended version of snakeoil3.

### TF2RL
An adapted version of the [tf2rl library](https://github.com/keiohta/tf2rl) by keiohta is included with the agents.
