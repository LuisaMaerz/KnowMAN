import yaml
import os

DEFAULT_Path = os.path.dirname(__file__)


class KnowManParameters:
    def __init__(self, config_file=DEFAULT_Path + "/../default_config.yaml"):
        self.dataset = {}
        self.model_params = {}
        self.training_setting = {}
        self.experiment_name = None

        with open(config_file) as f:
            config_content = yaml.safe_load(f)

        self.__dict__.update(config_content)

    def update_parameters(self, config_file):
        with open(config_file) as f:
            config_content = yaml.safe_load(f)

        self.__dict__.update(config_content)

    def get_config(self):
        return self.__dict__

