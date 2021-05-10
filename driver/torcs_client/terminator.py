import numpy as np

terminal_judge_start = 150
boring_speed = 3 # km/h, episode terminates if car is running slower than this limit

def custom_terminal(obs, curr_step):
    terminal = False

    angle = np.cos(obs["angle"])
    speed = np.array(obs["speedX"])

    # if (abs(track.any()) > 1 or abs(trackPos) > 1):
    # Episode is terminated if the car is out of track, too cruel to start
        # terminal = True

    if terminal_judge_start < curr_step:
        # Episode terminates if the agent is too slow
        if speed < boring_speed:
            terminal = True

    if angle < 0:
        # Episode is terminated if the agent runs backward
        terminal = True

    return terminal
