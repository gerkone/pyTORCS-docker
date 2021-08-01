import numpy as np
import argparse
import os
import h5py


def augment(dir):
    dataset_files = []
    for file in os.listdir(dir):
        if ".h5" in file:
            dataset_files.append(os.path.join(dir, file))

    for i, ep_file in zip(range(len(dataset_files)), dataset_files):
        print("loading episode data: {} - {}/{}".format(ep_file, i + 1, len(dataset_files)))

        dataset = h5py.File(ep_file, "r")

        dataset_dist = np.array(dataset.get("dist"))
        dataset_time = np.array(dataset.get("time"))

        action = np.array(dataset.get("action"))
        sensors = np.array(dataset.get("sensors"))

        # steer
        action = np.array([[-a[0], a[1]] for a in action])
        # angle
        for i, s in zip(range(sensors.shape[0]), sensors):
            sensors[i][3] = -s[3]
            sensors[i][3] = -s[4]
            sensors[i][9:28] =  np.flip(s[9:28][:], axis = 0)

        dataset_file = h5py.File("dataset/reversed_{}".format(ep_file.split("/")[-1]), "a")
        # telemetry
        dataset_file.create_dataset("dist", data = dataset_dist)
        dataset_file.create_dataset("time", data = dataset_time)
        # state
        dataset_file.create_dataset("sensors", data = sensors)
        # action
        dataset_file.create_dataset("action", data = action)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="estimate torcs sensor from image data")
    parser.add_argument("--dataset-dir", dest = "dataset_dir", help="path to the dataset directory", default="dataset/", type=str)

    args = parser.parse_args()

    augment(args.dataset_dir)
