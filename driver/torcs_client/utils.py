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


def start_container(name, verbose):
    # check if the container is already up
    id = subprocess.check_output(["docker", "ps", "-q", "--filter", "name=" + name])
    id = id.decode('utf-8')
    if len(id) == 0:
        # not yet started
        # get display from environment
        display = "unix" + os.environ["DISPLAY"]
        if verbose: print(bcolors.OKGREEN + "Starting TORCS container..." + bcolors.ENDC)
        subprocess.Popen(["nvidia-docker", "run", "-v", "/tmp/.X11-unix:/tmp/.X11-unix:ro",
            "-e", "DISPLAY=" + display, "-p", "3101:3101/udp", "--rm", "-t", "-d", "--name", name, "gerkone/vtorcs"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        while len(id) == 0:
            time.sleep(0.5)
            id = subprocess.check_output(["docker", "ps", "-q", "--filter", "ancestor=gerkone/torcs"])
            id = id.decode('utf-8')

        if verbose: print(bcolors.OKGREEN + "Container started with id " + id + bcolors.ENDC)
    else:
        if verbose: print(bcolors.OKGREEN +"Container " + id + " already running" + bcolors.ENDC)

def reset_torcs(torcs_on_docker, name, vision):
    if torcs_on_docker:
      subprocess.Popen(["docker", "exec", name, "sh", "kill.sh"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      if vision is True:
          subprocess.Popen(["docker", "exec", name, "sh", "start_vision.sh"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      else:
          subprocess.Popen(["docker", "exec", name, "sh", "start.sh"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
      subprocess.Popen(["pkill", "torcs"], stdout=subprocess.DEVNULL)
      if vision is True:
          subprocess.Popen(["torcs", "-nofuel", "-nodamage", "-nolaptime", "-vision"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      else:
          subprocess.Popen(["torcs", "-nofuel", "-nodamage", "-nolaptime"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      subprocess.Popen(["sh", "autostart.sh"], stdout=subprocess.DEVNULL)
