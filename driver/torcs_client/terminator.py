import numpy as np

terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
boring_speed = 1

def custom_terminal(obs, reward, time_step):
    terminal = False

    angle = np.cos(obs['angle'])
    speed = np.array(obs['speedX'])

    # if (abs(track.any()) > 1 or abs(trackPos) > 1):
    # Episode is terminated if the car is out of track, too cruel to start
        # terminal = True

    if terminal_judge_start < time_step:
        # Episode terminates if the agent is too slow
        if speed < boring_speed:
           terminal = True
        # Episode terminates if the progress of agent is small
        if reward < termination_limit_progress:
           terminal = True

    if angle < 0:
        # Episode is terminated if the agent runs backward
        terminal = True

    return terminal
