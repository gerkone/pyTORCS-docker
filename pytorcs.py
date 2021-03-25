import subprocess
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyTORCS: TORCS-based environment for simple autonomous driving simulations.")
    parser.add_argument("--config", help="Path to the yaml config file.", default="config/simulation.yaml", type=str)
    parser.add_argument("-v", "--verbose", help="Set verbose.", default=False, action="store_true")
    parser.add_argument("-c", "--console", help="Your console of choice.", default="konsole", type=str)

    args = parser.parse_args()
    # konsole, xterm
    execute = "-e"
    if args.console == "terminator" or args.console == "gnome-terminal":
        # terminator, gnome-terminal
        execute = "-x"
    if args.verbose:
        subprocess.Popen([args.console, execute, "python", "driver/launch.py", "--config", args.config, "--verbose"])
    else:
        subprocess.Popen([args.console, execute, "python", "driver/launch.py", "--config", args.config])
