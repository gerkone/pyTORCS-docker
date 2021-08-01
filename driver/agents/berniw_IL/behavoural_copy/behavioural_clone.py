import numpy as np
import argparse
import os
import h5py

from network import Agent

class BehaviouralClone:
    def __init__(self, dataset_dir, load, batch, epochs, lr = 5e-6):

        self.dataset_files = []

        for file in os.listdir(dataset_dir):
            if ".h5" in file:
                self.dataset_files.append(os.path.join(dataset_dir, file))

        self.state_dims = 28
        self.action_dims = 2
        self.batch = batch

        self.agent = Agent(state_dims = self.state_dims, action_dims = self.action_dims, lr = lr,
                load = load, save_dir = "BC_agent", batch_size = batch, epochs = epochs)

    def _prepare_data(self, filename):
        dataset = h5py.File(filename, "r")

        action = np.array(dataset.get("action"))
        sensors = np.array(dataset.get("sensors"))

        return action, sensors

    def batchify(self, iterable, n):
        l = iterable.shape[0]
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]


    def run(self):
        for i, ep_file in zip(range(len(self.dataset_files)), self.dataset_files):
            print("loading episode data: {} - {}/{}".format(ep_file, i + 1, len(self.dataset_files)))

            action, sensors = self._prepare_data(ep_file)

            self.agent.train(sensors, action)

            del action
            del sensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="estimate torcs sensor from image data")
    parser.add_argument("--dataset-dir", dest = "dataset_dir", help="path to the dataset directory", default="dataset/", type=str)
    parser.add_argument("--load", dest = "load_old", help="set to load saved model", default="False", action="store_true")
    parser.add_argument("--batch", dest = "batch", help="batch size", type = int, default = 32)
    parser.add_argument("--epochs", dest = "epochs", help="n of epochs", type = int, default = 5)

    args = parser.parse_args()

    bc = BehaviouralClone(args.dataset_dir, args.load_old, args.batch, args.epochs)

    bc.run()
