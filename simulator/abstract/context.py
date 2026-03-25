import inspect
import os
import random
import re
import socket
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Optional, Union

DEFAULT_F_TIME = 10

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


def sanitize_run_subdir_component(name: str) -> str:
    """Make a single path segment safe for common filesystems (macOS/Linux/Windows)."""
    s = str(name).replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    s = s.strip("._")
    return s or "unknown"


def refresh_run_output_paths(
    gpc: Optional[Union[Config, dict]] = None,
    *,
    micro_batch_num: Optional[int] = None,
) -> None:
    """
    Set ``RUN_OUTPUT_DIR`` and per-run artifact paths under:

        schedule_results/{schedule_name}_{model_name}_PP{pp}_mb{mb}/

    ``result.txt``, ``placement.txt``, and ``partition.txt`` are written there.

    Call **once per run** after ``SCHEDULE_METHOD``, ``PP_SIZE``/``DEVICE_NUM``,
    ``MODEL_NAME``, and micro-batch count are all finalized (e.g. after building
    ``TrainingConfig`` in ``__main__.py``). Calling this repeatedly with different
    parameters would otherwise create multiple output directories.
    """
    gpc = global_context if gpc is None else gpc
    sched = gpc["SCHEDULE_METHOD"].name
    model = gpc.get("MODEL_NAME", "default")
    pp = int(gpc["PP_SIZE"])
    mb = int(micro_batch_num) if micro_batch_num is not None else int(gpc["MICRO_BATCH_NUM"])
    sub = (
        f"{sanitize_run_subdir_component(sched)}_{sanitize_run_subdir_component(model)}"
        f"_PP{pp}_mb{mb}"
    )
    out_dir = os.path.join("schedule_results", sub)
    # Do not mkdir here: intermediate refreshes would create multiple sibling folders per run.
    # Directories are created when files are written (see save_to_file / save_partition).
    gpc["RUN_OUTPUT_DIR"] = out_dir
    gpc["SCHEDULING_PATH"] = os.path.join(out_dir, "result.txt")
    gpc["PLACEMENT_PATH"] = os.path.join(out_dir, "placement.txt")
    gpc["PARTITION_TXT_PATH"] = os.path.join(out_dir, "partition.txt")


global_context = Config.from_file("simulator/config.py")
# Per-run paths are set once by the entrypoint via refresh_run_output_paths() after
# schedule / device / model / micro-batch are finalized (see __main__.py).

def apply_stage_time_profile(model_name: str, seq_len: Union[int, None] = None) -> Config:
    """
    Apply profiled per-stage forward/backward times from `data.profiled_data.stage_time`
    into `global_context`, and infer `LAYER_NUM` automatically.

    Convention: stage_time[model][seq]["f"/"b"] contains per-layer/stage times with
    length equal to the model layer count. Therefore LAYER_NUM is inferred as len(times).
    """
    global global_context
    gpc = global_context

    from data.profiled_data import stage_time  # local import to avoid hard dependency at module import time
    supported_models = sorted(stage_time.keys())

    # Unknown model: fall back to default F/B/W times from `simulator/config.py` (164-178)
    if model_name not in stage_time:
        print(
            f"[WARN] Unknown model_name={model_name!r}. "
            f"Falling back to default timing config. "
            f"Supported profiled models: {supported_models}"
        )
        # Keep existing LAYER_NUM from config, only reset times like in config.py 164-178
        layer_num = gpc["LAYER_NUM"]
        f_time = DEFAULT_F_TIME

        gpc["MODEL_NAME"] = model_name
        gpc["F_TIME"] = f_time
        gpc["F_TIMES"] = [f_time] * layer_num
        gpc["B_TIMES"] = [f_time] * layer_num
        gpc["W_TIMES"] = [f_time] * layer_num
        if layer_num >= 1:
            gpc["F_TIMES"][-1] += f_time // 2
            gpc["B_TIMES"][-1] += 6

        # Reset non-transformer layer times as in config.py 170-178
        gpc["EMB_F_TIME"] = 0
        gpc["EMB_B_TIME"] = 0
        gpc["EMB_W_TIME"] = 0
        gpc["HEAD_F_TIME"] = 0
        gpc["HEAD_B_TIME"] = 0
        gpc["HEAD_W_TIME"] = 0
        gpc["CE_F_TIME"] = 0
        gpc["CE_B_TIME"] = 0
        gpc["CE_W_TIME"] = 0

        return gpc

    model_profiles = stage_time[model_name]
    if not model_profiles:
        raise KeyError(f"No stage_time profiles found for model_name={model_name!r}")

    if seq_len is None:
        preferred = gpc.get("SEQ_LEN", None)
        if preferred in model_profiles:
            seq_len = preferred
        else:
            seq_len = max(model_profiles.keys())

    if seq_len not in model_profiles:
        raise KeyError(
            f"Unknown seq_len={seq_len!r} for model_name={model_name!r}. "
            f"Available: {sorted(model_profiles.keys())}"
        )

    f_times = list(model_profiles[seq_len]["f"])
    b_times = list(model_profiles[seq_len]["b"])
    layer_num = len(f_times)

    arch = model_profiles[seq_len].get("arch", None)
    if len(f_times) != len(b_times):
        raise ValueError(
            f"stage_time length mismatch for {model_name!r} seq_len={seq_len}: "
            f"len(f)={len(f_times)} != len(b)={len(b_times)}"
        )
    if len(f_times) < 1:
        raise ValueError(f"stage_time too short for {model_name!r} seq_len={seq_len}: {len(f_times)}")

    if arch is not None and len(arch) != layer_num:
        raise ValueError(
            f"stage_time length mismatch for {model_name!r} seq_len={seq_len}: "
            f"len(arch)={len(arch)} != len(f)={layer_num}"
        )

    gpc["MODEL_NAME"] = model_name
    gpc["SEQ_LEN"] = seq_len
    gpc["LAYER_NUM"] = layer_num
    gpc["ARCH"] = list(arch) if arch is not None else None
    gpc["F_TIMES"] = f_times
    gpc["B_TIMES"] = b_times
    # Keep shapes consistent; actual W_TIMES may be overwritten later (e.g. when bwd_split=True).
    gpc["W_TIMES"] = [0 for _ in range(layer_num)]

    return gpc


def apply_device_config(device_num: int) -> Config:
    """
    Update DEVICE_NUM and related derived fields in `global_context`.
    This mirrors the logic in `simulator/config.py` so that the number of
    pipeline stages/devices can be controlled from the entrypoint.
    """
    global global_context
    gpc = global_context

    gpc["DEVICE_NUM"] = device_num
    gpc["PP_SIZE"] = device_num

    # Recompute CHUNK_NUM based on SCHEDULE_METHOD (similar policy as config.py).
    schedule_method = gpc["SCHEDULE_METHOD"]
    chunk_num = gpc.get("CHUNK_NUM", 1)
    from simulator.abstract.variables import Schedule  # local import to avoid cycles at module import
    if schedule_method == Schedule.STANDARD_INTERLEAVED:
        chunk_num = gpc["LAYER_NUM"] // device_num
    elif schedule_method in (Schedule.STANDARD_ZBH, Schedule.STANDARD_1F1B, Schedule.STANDARD_AFAB):
        chunk_num = 1
    gpc["CHUNK_NUM"] = chunk_num

    # Recompute stage-related stats.
    stage_num = int(device_num * chunk_num)
    gpc["STAGE_NUM"] = stage_num
    gpc["MAX_ACTIVATION_COUNTS"] = int(stage_num * 2)

    # Recompute schedule/save-file paths so they stay consistent with new DEVICE_NUM.
    try:
        heter = gpc["HETER_DEVICE"]
        vocab = gpc["VOCAB_SIZE"]
        lnum = gpc["LAYER_NUM"]
        slen = gpc["SEQ_LEN"]
        hsize = gpc["HIDDEN_SIZE"]
        mbnum = gpc["MICRO_BATCH_NUM"]
        pp = gpc["PP_SIZE"]
        tp = gpc["TP_SIZE"]
        zr = gpc["ZERO_SIZE"]
        ch = gpc["CHUNK_NUM"]
        sm = gpc["SCHEDULE_METHOD"]
        sp = gpc["STAGE_PLACEMENT"]
        w = gpc["SPLIT_BACKPROP"]
        lw = gpc["LAYERWISE"]
        od = gpc["OVERLAP_DEGREE"]

    except Exception:
        # Best-effort; if anything is missing we just skip path recomputation.
        pass

    return gpc

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