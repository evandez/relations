__version__ = "1.1.3.post1"

# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba.mamba_ssm.modules.mamba_simple import Mamba
from mamba.mamba_ssm.ops.selective_scan_interface import (
    mamba_inner_fn,
    selective_scan_fn,
)
