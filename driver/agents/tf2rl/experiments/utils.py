import os
import h5py
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import animation


def load_expert_traj(dataset_dir, max_steps = 2500):
    dataset_files = []

    for file in os.listdir(dataset_dir):
        if ".h5" in file:
            dataset_files.append(os.path.join(dataset_dir, file))


    curr_ep = 0

    expert_traj = {}
    expert_traj["state"] = []
    expert_traj["state_new"] = []
    expert_traj["action"] = []

    while curr_ep < len(dataset_files):
        ep_file = dataset_files[curr_ep]

        print("Loading expert trjectory: {} - {}/{}".format(ep_file, curr_ep + 1, len(dataset_files)))

        dataset = h5py.File(ep_file, "r")

        action = np.array(dataset.get("action"))
        state = np.array(dataset.get("sensors"))

        for el in range(min(max_steps, len(action) - 1)):
            next = el + 1

            expert_traj["state"].append(state[el])
            expert_traj["action"].append(action[el])
            expert_traj["state_new"].append(state[next])

        curr_ep += 1

    expert_traj["state"] = np.array(expert_traj["state"])
    expert_traj["state_new"] = np.array(expert_traj["state_new"])
    expert_traj["action"] = np.array(expert_traj["action"])

    return expert_traj

def save_path(samples, filename):
    joblib.dump(samples, filename, compress=3)


def get_filenames(dirname, n_path=None):
    import re
    itr_reg = re.compile(
        r"step_(?P<step>[0-9]+)_epi_(?P<episodes>[0-9]+)_return_(-?)(?P<return_u>[0-9]+).(?P<return_l>[0-9]+).pkl")

    itr_files = []
    for _, filename in enumerate(os.listdir(dirname)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('step')
            itr_files.append((itr_count, filename))

    n_path = n_path if n_path is not None else len(itr_files)
    itr_files = sorted(itr_files, key=lambda x: int(
        x[0]), reverse=True)[:n_path]
    filenames = []
    for itr_file_and_count in itr_files:
        filenames.append(os.path.join(dirname, itr_file_and_count[1]))
    return filenames


def load_trajectories(filenames, max_steps=None):
    assert len(filenames) > 0
    paths = []
    for filename in filenames:
        paths.append(joblib.load(filename))

    def get_obs_and_act(path):
        obses = path['obs'][:-1]
        next_obses = path['obs'][1:]
        actions = path['act'][:-1]
        if max_steps is not None:
            return obses[:max_steps], next_obses[:max_steps], actions[:max_steps-1]
        else:
            return obses, next_obses, actions

    for i, path in enumerate(paths):
        if i == 0:
            obses, next_obses, acts = get_obs_and_act(path)
        else:
            obs, next_obs, act = get_obs_and_act(path)
            obses = np.vstack((obs, obses))
            next_obses = np.vstack((next_obs, next_obses))
            acts = np.vstack((act, acts))
    return {'obses': obses, 'next_obses': next_obses, 'acts': acts}


def frames_to_gif(frames, prefix, save_dir, interval=50, fps=30):
    """
    Convert frames to gif file
    """
    assert len(frames) > 0
    plt.figure(figsize=(frames[0].shape[1] / 72.,
                        frames[0].shape[0] / 72.), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    # TODO: interval should be 1000 / fps ?
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=interval)
    output_path = "{}/{}.gif".format(save_dir, prefix)
    anim.save(output_path, writer='imagemagick', fps=fps)
