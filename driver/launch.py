import os
import argparse
import yaml
import importlib.util

class Launch:
    def __init__(self, args):
        self.verbose = args.verbose
        self.config_path = args.config

        self.config = self._load_config()

        self.entrypoint = self._entrypoint()

    def _entrypoint(self):
        """
        import custom run function from module path
        """
        spec = importlib.util.spec_from_file_location(self.mod_name, self.run_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, self.mod_name)

    def _load_config(self):
        """
        load yaml config
        """
        with open(self.config_path, 'r') as stream:
            try:
                conf = yaml.safe_load(stream)
                self.mod_name = conf["mod_name"]
                self.run_path = conf["run_path"]
                self.image_name = conf["image_name"]

                self.hyperparams = conf["hyperparams"]
                self.sensors = conf["sensors"]
                self.img_width = conf["img_width"]
                self.img_height = conf["img_height"]
            except yaml.YAMLError as exc:
                print(exc)

    def run(self):
        self.entrypoint(verbose = self.verbose, hyperparams = self.hyperparams, sensors = self.sensors)


if __name__ == "__main__":
    # set tensorflow to train with GPU 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="pyTORCS: TORCS-based environment for simple autonomous driving simulations.")
    parser.add_argument("--config", help="Path to the yaml config file.", default="config/simulation.yaml", type=str)
    parser.add_argument("--verbose", help="Set verbose.", default=False, action="store_true")

    args = parser.parse_args()

    pytorcs = Launch(args)
    # launch system
    pytorcs.run()
