"""Run the maintained tests/test_sdpa.py on ONE small config, functional mode ON,
to ground-truth whether the non-causal flash template is numerically correct here.
"""
import sys, os, importlib.util, torch
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
from torch.nn.attention import sdpa_kernel, SDPBackend

spec = importlib.util.spec_from_file_location("tsdpa", "/workspace/PyTorchSim/tests/test_sdpa.py")
tsdpa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsdpa)

with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    tsdpa.test_sdpa(
        tsdpa.device,
        n_batch_list=[1], n_head_list=[5], n_token_list=[64], head_dim_list=[128],
        is_causal=False,
    )
