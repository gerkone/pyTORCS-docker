import subprocess
import time
import os
import re

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def start_container(image_name, verbose, port):
    # check if the container is already up
    container_id = subprocess.check_output(["docker", "ps", "-q", "--filter", "ancestor=" + image_name]).decode('utf-8')
    if len(container_id) == 0:
        # not yet started
        # get display from environment
        display = "unix" + os.environ["DISPLAY"]
        if verbose: print(bcolors.OKGREEN + "Starting TORCS container..." + bcolors.ENDC)
        subprocess.Popen(["nvidia-docker", "run", "-v", "/tmp/.X11-unix:/tmp/.X11-unix:ro",
            "-e", "DISPLAY=" + display, "-p", "{p}:{p}/udp".format(p = port), "--rm", "-t", "-d", image_name])
        time.sleep(0.5)
        while len(container_id) == 0:
            time.sleep(0.5)
            container_id = subprocess.check_output(["docker", "ps", "-q", "--filter", "ancestor=" + image_name]).decode('utf-8')

        if verbose: print(bcolors.OKGREEN + "Container started with id " + container_id + bcolors.ENDC)
    else:
        if verbose: print(bcolors.OKGREEN +"Container " + container_id + " already running" + bcolors.ENDC)

    return re.sub("[^a-zA-Z0-9 -]", "", container_id)

def reset_torcs(container_id, vision, kill = False):
    command = []

    if kill:
        kill_torcs(container_id)

    if container_id != "0":
        command.extend(["docker", "exec", container_id, "torcs"])

    command.extend(["torcs", "-nofuel", "-nodamage", "-nolaptime"])

    if vision is True:
        command.extend("-vision")

    time.sleep(1)
    subprocess.Popen(command)

def kill_torcs(container_id):
    command = []

    if container_id != "0":
        command.extend(["docker", "exec", container_id])
    command.extend(["pkill", "torcs"])
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            if self.verbose: print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]
