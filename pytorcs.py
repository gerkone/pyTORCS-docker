import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyTORCS: TORCS-based environment for simple autonomous driving simulations.")
    parser.add_argument("--config", help="Path to the yaml config file.", default="config/sim_ddpg.yaml", type=str)
    parser.add_argument("-v", "--verbose", help="Set verbose.", default=False, action="store_true")
    parser.add_argument("--privileged", help="Set privileged. Attempts to solve the NVML runtime docker issue.", default=False, action="store_true")
    parser.add_argument("-c", "--console", help="Your console of choice.", default="", type=str)

    args = parser.parse_args()
    launch_command = []
    if args.console != "":
        # konsole, xterm
        execute = "-e"
        if args.console == "terminator" or args.console == "gnome-terminal":
            # terminator, gnome-terminal
            execute = "-x"
        launch_command.extend([args.console, execute])

    launch_command.extend(["python", "driver/launch.py", "--config", args.config])

    if args.verbose == True:
        launch_command.append("--verbose")

    if args.privileged == True:
        launch_command.append("--privileged")

    if args.console != "":
        # detach
        subprocess.Popen(launch_command)
    else:
        # synchronous
        subprocess.call(launch_command)
