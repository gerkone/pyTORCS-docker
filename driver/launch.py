import os
import argparse
import yaml
import importlib.util

class Launch:
    """
    Load config files and run module and launch
    """
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
                self.algo_name = conf["algo_name"]
                self.algo_path = conf["algo_path"]

                self.image_name = conf["image_name"]

                self.hyperparams = conf["hyperparams"]
                self.sensors = conf["sensors"]
                self.training = conf["training"]

                try:
                    self.stack_depth = conf["stack_depth"]
                    self.img_width = conf["img_width"]
                    self.img_height = conf["img_height"]
                except:
                    self.stack_depth = None
                    self.img_width = None
                    self.img_height = None

            except yaml.YAMLError as exc:
                print(exc)

    def run(self):
        self.entrypoint(verbose = self.verbose, hyperparams = self.hyperparams, sensors = self.sensors,
                training = self.training, algo_name = self.algo_name, algo_path = self.algo_path, image_name = self.image_name,
                stack_depth = self.stack_depth, img_width = self.img_width, img_height = self.img_height)

if __name__ == "__main__":
    try:
        print("pyTORCS: TORCS-based environment for simple autonomous driving simulations.")
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
    except Exception:
        import traceback
        traceback.print_exc()
        input("pytorcs crashed")
