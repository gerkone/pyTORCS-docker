import subprocess
import argparse

from driver.test_torcs import main


parser = argparse.ArgumentParser(description="pyTORCS: TORCS-based environment for simple autonomous driving simulations.")

parser.add_argument("-d", "--nodocker", help="Set to run the system on host (requires manual vtorcs installation).", default=False, type=bool)

parser.add_argument("-v", "--verbose", help="Set verbose.", default=False, type=bool)

args = parser.parse_args()

# compose = ["docker-compose", "up"]

run_on_docker = not args.nodocker

# if run_on_docker:
    # compose.append("torcs")
    # if not args.tf:
    #     compose.append("driver")
    # subprocess.Popen(compose)
# else:

main(torcs_on_docker = run_on_docker, vision = True, verbose = args.verbose)
