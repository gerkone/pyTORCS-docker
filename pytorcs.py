import subprocess
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyTORCS: TORCS-based environment for simple autonomous driving simulations.")
    parser.add_argument("-c", "--config", help="Path to the yaml config file.", default="config/simulation.yaml", type=str)
    parser.add_argument("-v", "--verbose", help="Set verbose.", default=False, action="store_true")

    args = parser.parse_args()

    if args.verbose:
        subprocess.Popen(["python", "driver/launch.py", "--config", args.config, "--verbose"])
    else:
        subprocess.Popen(["python", "driver/launch.py", "--config", args.config])
