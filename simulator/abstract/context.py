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

global_context = Config.from_file("simulator/config.py")

def flush_fbw_time(bwd_split, ideal_case):
    global global_context
    gpc = global_context
    gpc["F_TIME"] = 10
    gpc["F_TIMES"] = [gpc["F_TIME"]] * gpc["LAYER_NUM"]
    gpc["B_TIMES"] = [gpc["F_TIME"]] * gpc["LAYER_NUM"]
    gpc["W_TIMES"] = [gpc["F_TIME"]] * gpc["LAYER_NUM"]

    if not ideal_case:
        if gpc["GEMMA"]:
            try:
                from data.profiled_data import profiled_data
                ratios = profiled_data["GEMMA"][gpc["HIDDEN_SIZE"]][gpc["SEQ_LEN"]][gpc["VOCAB_SIZE"]]
                [tf_tf, tb_tf, tw_tf, _, _, _, hf_tf, hb_tf, hw_tf] = [round(r, 1) for r in ratios]
                gpc["B_TIMES"] = [t*(tb_tf+tw_tf) for i,t in enumerate(gpc["F_TIMES"])]
                gpc["HEAD_F_TIME"] = gpc["F_TIME"] * hf_tf
                gpc["HEAD_B_TIME"] = gpc["F_TIME"] * (hb_tf + hw_tf)
                if bwd_split:
                    gpc["B_TIMES"] = [t*tb_tf for i,t in enumerate(gpc["F_TIMES"])]
                    gpc["W_TIMES"] = [t*tw_tf for i,t in enumerate(gpc["F_TIMES"])]
                    gpc["HEAD_B_TIME"] = gpc["F_TIME"] * hb_tf
                    gpc["HEAD_W_TIME"] = gpc["F_TIME"] * hw_tf
            except:
                print("----- No profiled data! Use predefined ratios. -----")

        if gpc["DEEPSEEK"]:
            try:
                from data.profiled_data import profiled_data
                ratios = profiled_data["DEEPSEEK"][gpc["HIDDEN_SIZE"]][gpc["SEQ_LEN"]][gpc["VOCAB_SIZE"]]
                [tf_tf, tb_tf, tw_tf, mf_tf, mb_tf, mw_tf, hf_tf, hb_tf, hw_tf] = [round(r, 1) for r in ratios]
                gpc["B_TIMES"] = [t*(mb_tf+mw_tf) if i >= gpc["LAYER_NUM"]//gpc["PP_SIZE"] - 1  else t * (tb_tf+tw_tf) for i,t in enumerate(gpc["F_TIMES"])]
                gpc["HEAD_F_TIME"] = gpc["F_TIME"] * hw_tf
                gpc["HEAD_B_TIME"] = gpc["F_TIME"] * (hb_tf+hw_tf)
                if bwd_split:
                    if tw_tf == 0:
                        tw_tf = 0.2
                        tb_tf -= tw_tf
                    gpc["B_TIMES"] = [t*mb_tf if i >= gpc["LAYER_NUM"]//gpc["PP_SIZE"] - 1  else t * tb_tf for i,t in enumerate(gpc["F_TIMES"])]
                    gpc["W_TIMES"] = [t*mw_tf if i >= gpc["LAYER_NUM"]//gpc["PP_SIZE"] - 1  else t * tw_tf for i,t in enumerate(gpc["F_TIMES"])]
                    gpc["HEAD_B_TIME"] = gpc["F_TIME"] * hb_tf
                    gpc["HEAD_W_TIME"] = gpc["F_TIME"] * hw_tf
                gpc["F_TIMES"] = [t*mf_tf if i >= gpc["LAYER_NUM"]//gpc["PP_SIZE"] - 1 else t for i,t in enumerate(gpc["F_TIMES"])]
            except:
                print("----- No profiled data! Use predefined ratios. -----")

        if gpc["NEMOTRONH"]:
            diff = 3 * gpc["N_SCALE"]
            try:
                from data.profiled_data import profiled_data
                ratios = profiled_data["NEMOTRONH"][gpc["HIDDEN_SIZE"]][gpc["SEQ_LEN"]][gpc["VOCAB_SIZE"]]
                [tf_mf, tb_mf, tw_mf, mf_mf, mb_mf, mw_mf, hf_mf, hb_mf, hw_mf] = [round(r, 1) for r in ratios]
                gpc["B_TIMES"] = [t*(tb_mf+tw_mf) if (i+1)%diff==0  else t * mb_mf for i,t in enumerate(gpc["F_TIMES"])]
                gpc["HEAD_F_TIME"] = gpc["F_TIME"] * hf_mf
                gpc["HEAD_B_TIME"] = gpc["F_TIME"] * (hb_mf + hw_mf)
                if bwd_split:
                    gpc["B_TIMES"] = [t*tb_mf if (i+1)%diff==0  else t * (mb_mf-0.1) for i,t in enumerate(gpc["F_TIMES"])]
                    gpc["W_TIMES"] = [t*tw_mf if (i+1)%diff==0  else t * 0.1 for i,t in enumerate(gpc["F_TIMES"])]
                    gpc["HEAD_B_TIME"] = gpc["F_TIME"] * hb_mf
                    gpc["HEAD_W_TIME"] = gpc["F_TIME"] * hw_mf
                gpc["F_TIMES"] = [t*tf_mf if (i+1)%diff==0 else t for i,t in enumerate(gpc["F_TIMES"])]
            except:
                print("----- No profiled data! Use predefined ratios. -----")

        if gpc["VARYLEN"]:
            diff = 12
            from data.profiled_data import profiled_data
            ratios = profiled_data["NEMOTRONH"][gpc["HIDDEN_SIZE"]][gpc["SEQ_LEN"]][gpc["VOCAB_SIZE"]]
            [tf_mf, tb_mf, tw_mf, mf_mf, mb_mf, mw_mf, hf_mf, hb_mf, hw_mf] = [round(r, 1) for r in ratios]
            gpc["B_TIMES"] = [t*(tb_mf+tw_mf) if (i+1)%diff==0  else t * mb_mf for i,t in enumerate(gpc["F_TIMES"])]
            gpc["HEAD_F_TIME"] = gpc["F_TIME"] * hf_mf
            gpc["HEAD_B_TIME"] = gpc["F_TIME"] * (hb_mf + hw_mf)
            if bwd_split:
                gpc["B_TIMES"] = [t*tb_mf if (i+1)%diff==0  else t * (mb_mf-0.1) for i,t in enumerate(gpc["F_TIMES"])]
                gpc["W_TIMES"] = [t*tw_mf if (i+1)%diff==0  else t * 0.1 for i,t in enumerate(gpc["F_TIMES"])]
                gpc["HEAD_B_TIME"] = gpc["F_TIME"] * hb_mf
                gpc["HEAD_W_TIME"] = gpc["F_TIME"] * hw_mf
            gpc["F_TIMES"] = [t*tf_mf if (i+1)%diff==0 else t for i,t in enumerate(gpc["F_TIMES"])]
            print("------ Test vary length sequence. -----")
    # print(gpc["F_TIMES"],gpc["B_TIMES"],gpc["W_TIMES"])
    else:
        gpc["EMB_F_TIME"] = 0
        gpc["EMB_B_TIME"] = 0
        gpc["EMB_W_TIME"] = 0
        gpc["HEAD_F_TIME"] = 10
        gpc["HEAD_B_TIME"] = 2
        gpc["HEAD_W_TIME"] = 2
        gpc["CE_F_TIME"] = 0
        gpc["CE_B_TIME"] = 0
        gpc["CE_W_TIME"] = 0

if __name__ == "__main__":
    c = Config()
    c = Config.from_file("simulator/config.py")
    print(c)
    c['K'] = 1
    print(c["K"])