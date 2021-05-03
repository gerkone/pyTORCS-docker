import subprocess
import time
import os
import re
import numpy as np
import cv2

class SimpleLogger:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def info(str):
        print(SimpleLogger.OKBLUE + "[INFO]: " + str + SimpleLogger.ENDC)
    @staticmethod
    def alert(str):
        print(SimpleLogger.WARNING + "[WARN]: " + str + SimpleLogger.ENDC)
    @staticmethod
    def error(str):
        print(SimpleLogger.FAIL + "[ERR]: " + str + SimpleLogger.ENDC)
    @staticmethod
    def training(str, loss):
        print("--> " + str + SimpleLogger.OKGREEN + "Loss {:.3f}".format(loss) + SimpleLogger.ENDC)
    @staticmethod
    def separator(columns):
        print()
        print("-" * int(columns))
        print()

def start_container(image_name, verbose, port):
    # check if the container is already up
    container_id = subprocess.check_output(["docker", "ps", "-q", "--filter", "ancestor=" + image_name]).decode('utf-8')
    if len(container_id) == 0:
        # not yet started
        # get display from environment
        display = "unix" + os.environ["DISPLAY"]
        torcs_config_dir = os.path.join(os.getcwd(), "torcs/configs/config")
        if verbose: SimpleLogger.info("Starting TORCS container...")
        subprocess.Popen(["nvidia-docker", "run", "--ipc=host",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:ro",
            "-v", "{}:/usr/local/share/games/torcs/config:ro".format(torcs_config_dir),
            "-v", "{}:/root/.torcs/config:ro".format(torcs_config_dir),
            "-e", "DISPLAY=" + display,
            "-p", "{p}:{p}/udp".format(p = port),
            "--rm", "-t",
            "-d", image_name], stdout=subprocess.DEVNULL)
        time.sleep(0.5)
        while len(container_id) == 0:
            time.sleep(0.5)
            container_id = subprocess.check_output(["docker", "ps", "-q", "--filter", "ancestor=" + image_name]).decode('utf-8')
        container_id = re.sub("[^a-zA-Z0-9 -]", "", container_id)
        if verbose: SimpleLogger.info("Container started with id {}".format(container_id))
    else:
        container_id = re.sub("[^a-zA-Z0-9 -]", "", container_id)
        if verbose: SimpleLogger.info("Container {} already running".format(container_id))

    return container_id

def reset_torcs(container_id, vision, kill = False):
    command = []

    if kill:
        kill_torcs(container_id)

    if container_id != "0":
        command.extend(["docker", "exec", container_id])

    command.extend(["torcs", "-nofuel", "-nodamage", "-nolaptime"])

    if vision is True:
        command.extend(["-vision"])

    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def kill_torcs(container_id):
    command = []

    if container_id != "0":
        command.extend(["docker", "exec", container_id])
    command.extend(["pkill", "torcs"])
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def kill_container(container_id):
    command = []

    if container_id != "0":
        command.extend(["docker", "kill", container_id])
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def destringify(s):
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            if self.verbose: SimpleLogger.alert("Could not find a value in {}".format(s))
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]

def raw_to_rgb(img_buf, img_width, img_height):

    img = np.array(img_buf.reshape((img_height, img_width, 3)))
    img = np.flip(img, axis = 0)

    return img

def resize_frame(img, dest_width, dest_height):
    return cv2.resize(img, dsize=(dest_width, dest_height), interpolation=cv2.INTER_CUBIC)
