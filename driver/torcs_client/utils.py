import subprocess
import time
import os

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


def start_container(image_name, verbose):
    # check if the container is already up
    container_id = subprocess.check_output(["docker", "ps", "-q", "--filter", "ancestor=" + image_name])
    container_id = container_id.decode('utf-8')
    if len(container_id) == 0:
        # not yet started
        # get display from environment
        display = "unix" + os.environ["DISPLAY"]
        if verbose: print(bcolors.OKGREEN + "Starting TORCS container..." + bcolors.ENDC)
        subprocess.Popen(["nvidia-docker", "run", "-v", "/tmp/.X11-unix:/tmp/.X11-unix:ro",
            "-e", "DISPLAY=" + display, "-p", "3101:3101/udp", "--rm", "-t", "-d", "gerkone/vtorcs"])
        time.sleep(0.5)
        while len(container_id) == 0:
            time.sleep(0.5)
            container_id = subprocess.check_output(["docker", "ps", "-q", "--filter", "ancestor=" + image_name])
            container_id = container_id.decode('utf-8')

        if verbose: print(bcolors.OKGREEN + "Container started with container_id " + container_id + bcolors.ENDC)
    else:
        if verbose: print(bcolors.OKGREEN +"Container " + container_id + " already running" + bcolors.ENDC)

    return container_id

def reset_torcs(container_id, vision):
    if container_id != "0":
        print(container_id)
        subprocess.Popen(["docker", "exec", container_id, "sh", "kill.sh"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if vision is True:
          subprocess.Popen(["docker", "exec", container_id, "sh", "start_vision.sh"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
          subprocess.Popen(["docker", "exec", container_id, "sh", "start.sh"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(["pkill", "torcs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if vision is True:
          subprocess.Popen(["torcs", "-nofuel", "-nodamage", "-nolaptime", "-vision"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
          subprocess.Popen(["torcs", "-nofuel", "-nodamage", "-nolaptime"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # wait for window to pop up
        time.sleep(0.5)
        os.chdir(os.path.dirname(__file__))
        subprocess.Popen([os.getcwd() + "/autostart.sh"])
