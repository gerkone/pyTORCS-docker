import numpy as np

terminal_judge_start = 150
boring_speed = 0.5 # km/h, episode terminates if car is running slower than this limit
max_damage = 300
out_max = 0

def custom_terminal(obs, curr_step):

    terminal = False

    angle = np.cos(obs["angle"])
    speed = obs["speedX"]
    damage = obs["damage"]
    track_pos = obs["trackPos"]

    if np.abs(track_pos) > 1:
        if custom_terminal.first_out == -1:
            custom_terminal.first_out = curr_step
        # terminated if the car is out of track for too long
        if curr_step > custom_terminal.first_out + out_max:
            terminal = True
    else:
        # reset if car is back to track - works for initialization and reset
        custom_terminal.first_out = -1

    if damage > max_damage:
        terminal = True

    if terminal_judge_start < curr_step:
        # Episode terminates if the agent is too slow
        if speed < boring_speed:
            terminal = True

    if angle < 0:
        # Episode is terminated if the agent runs backward
        terminal = True

    try:
        if obs["distRaced"] > obs["trackLen"] * 2 + 10:
            terminal = True
    except Exception:
        pass

    return terminal
