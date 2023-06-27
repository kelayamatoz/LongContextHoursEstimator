# %%
TRUST_REMOTE_CODE = True
import transformers
import torch
from accelerate import init_empty_weights
from deepspeed.profiling.flops_profiler import FlopsProfiler
from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
from deepspeed.accelerator import set_accelerator, get_accelerator

K = 1024

class FakeGPU(CUDA_Accelerator):
    
    def synchronize(self):
        return

def arange_patched(*args, **kwargs):
    if len(args) == 3:
        start, end, step_size = args
        return torch.ones((end - start) // step_size, device='meta')
    elif len(args) == 1:
        return torch.ones(args[0], device='meta')
    elif len(args) == 2:
        start, end = args
        return torch.ones((end - start) // 1, device='meta')
    else:
        import pdb; pdb.set_trace()

torch.arange = arange_patched

set_accelerator(FakeGPU())

with init_empty_weights():
    falcon40b = "tiiuae/falcon-40b"
    # falcon40b = "mosaicml/mpt-30b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(falcon40b, trust_remote_code=TRUST_REMOTE_CODE)
    config = transformers.AutoConfig.from_pretrained(falcon40b, trust_remote_code=TRUST_REMOTE_CODE)
    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=TRUST_REMOTE_CODE)

    def ds_str_parse(s: str):
        return float(s.split(" ")[0])

    def get_flops_and_params(seq_len: int):
        profiler = FlopsProfiler(model)
        profiler.start_profile()
        batch_size = 1
        sequence_length = seq_len
        input_ids = torch.ones((batch_size, sequence_length), device='meta').long()
        profiler = FlopsProfiler(model)
        profiler.start_profile()
        out = model(input_ids)
        flops = profiler.get_total_flops(as_string=True)
        params = profiler.get_total_params(as_string=True)
        print(f"Total flops = {flops}, params = {params}")
        profiler.end_profile()
        flops_in_G = ds_str_parse(flops)
        params_in_M = ds_str_parse(params)
        return (flops_in_G, params_in_M)

# %%
(flops_2k, params_2k )= get_flops_and_params(2 * K)
(flops_16k, params_16k) = get_flops_and_params(16 * K)    
(flops_32k, params_32k) = get_flops_and_params(32 * K)
# %%
print(f"32K flops budget / 2K flops budget: {flops_32k // flops_2k}X")
print(f"16K flops budget / 2K flops budget: {flops_16k // flops_2k}X")