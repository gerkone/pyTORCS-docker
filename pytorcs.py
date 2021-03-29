import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyTORCS: TORCS-based environment for simple autonomous driving simulations.")
    parser.add_argument("--config", help="Path to the yaml config file.", default="config/simulation.yaml", type=str)
    parser.add_argument("-v", "--verbose", help="Set verbose.", default=False, action="store_true")
    parser.add_argument("-c", "--console", help="Your console of choice.", default="konsole", type=str)
    parser.add_argument("--noconsole", help="Avoid opening a new console (does not work properly).", default=False, action="store_true")

    args = parser.parse_args()
    if not args.noconsole:
        # konsole, xterm
        execute = "-e"
        if args.console == "terminator" or args.console == "gnome-terminal":
            # terminator, gnome-terminal
            execute = "-x"
        console = [args.console, execute]
    else:
        console = []
    if args.verbose:
        command = console + ["python", "driver/launch.py", "--config", args.config, "--verbose"]
    else:
        command = console + ["python", "driver/launch.py", "--config", args.config]
    subprocess.Popen(command)
