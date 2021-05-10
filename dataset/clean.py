import numpy as np
import h5py
import os

def clean(dataset_files):
    for i, ep_file in zip(range(len(dataset_files)), dataset_files):
        print("loading episode data: {} - {}/{}".format(ep_file, i + 1, len(dataset_files)))

        dataset = h5py.File(ep_file, "r+")
        images = np.array(dataset.get("img"))
        sensors = np.array(dataset.get("sensors"))

        marked = []
        initial_shape = (images.shape[0], sensors.shape[0])
        for i in range(min(images.shape[0], sensors.shape[0]) - 1):
            # remove any external measurement
            # when the car is outside the track boundaries all rangefinder values are negative
            if sensors[i][3] < 0:
                marked.append(i)

        sensors = np.delete(sensors, marked, axis = 0)
        images = np.delete(images, marked, axis = 0)

        final_shape = (images.shape[0], sensors.shape[0])

        del dataset["img"]
        del dataset["sensors"]
        dataset.create_dataset("img", data = images, compression="gzip", chunks=True)
        dataset.create_dataset("sensors", data = sensors, compression="gzip", chunks=True)
        dataset.close()
        print("-> {} cleaned: from {} to {}".format(ep_file, initial_shape, final_shape))

if __name__ == "__main__":
    files = []
    dataset_dir = "."

    for file in os.listdir(dataset_dir):
        if ".h5" in file:
            files.append(os.path.join(dataset_dir, file))

    print("removing negative rangefinder values ( car out of track )")
    clean(files)
