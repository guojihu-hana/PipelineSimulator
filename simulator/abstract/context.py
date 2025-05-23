import inspect
import random
import socket
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Union
import sys
sys.path.append(".")
class Config(dict):
    """This is a wrapper class for dict objects so that values of which can be
    accessed as attributes.

    Args:
        config (dict): The dict object to be wrapped.
    """

    def __init__(self, config: dict = None):  # pylint: disable=W0231
        if config is not None:
            for k, v in config.items():
                self._add_item(k, v)

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, key):
        try:
            value = super().__getitem__(key)
            return value
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        super().__setitem__(key, value)

    def _add_item(self, key, value):
        if isinstance(value, dict):
            self.__setattr__(key, Config(value))
        else:
            self.__setattr__(key, value)

    def update(self, config):
        assert isinstance(config, (Config, dict)), "can only update dictionary or Config objects."
        for k, v in config.items():
            self._add_item(k, v)
        return self

    @staticmethod
    def from_file(filename: str):
        """Reads a python file and constructs a corresponding :class:`Config` object.

        Args:
            filename (str): Name of the file to construct the return object.

        Returns:
            :class:`Config`: A :class:`Config` object constructed with information in the file.

        Raises:
            AssertionError: Raises an AssertionError if the file does not exist, or the file is not .py file
        """

        # check config path
        if isinstance(filename, str):
            filepath = Path(filename).absolute()
        elif isinstance(filename, Path):
            filepath = filename.absolute()

        assert filepath.exists(), f"{filename} is not found, please check your configuration path"

        # check extension
        extension = filepath.suffix
        assert extension == ".py", "only .py files are supported"

        # import the config as module
        remove_path = False
        if filepath.parent not in sys.path:
            sys.path.insert(0, (filepath))
            remove_path = True

        module_name = filepath.stem
        source_file = SourceFileLoader(fullname=str(module_name), path=str(filepath))
        module = source_file.load_module()  # pylint: disable=W4902,E1120,W1505

        # load into config
        config = Config()

        for k, v in module.__dict__.items():
            if k.startswith("__") or inspect.ismodule(v) or inspect.isclass(v):
                continue
            else:
                config._add_item(k, v)

        # remove module
        del sys.modules[module_name]
        if remove_path:
            sys.path.pop(0)

        return config

global_context = Config.from_file("/Users/hanayukino/Code/PipelineSimulator/simulator/config.py")

if __name__ == "__main__":
    c = Config()
    c = Config.from_file("/Users/hanayukino/Code/PipelineSimulator/simulator/config.py")
    print(c)
    c['K'] = 1
    print(c["K"])