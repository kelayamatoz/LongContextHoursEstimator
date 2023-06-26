# %%
TRUST_REMOTE_CODE = True
import transformers
import torch
from accelerate import init_empty_weights
from deepspeed.profiling.flops_profiler import FlopsProfiler

with init_empty_weights():
    falcon40b = "tiiuae/falcon-40b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(falcon40b, trust_remote_code=TRUST_REMOTE_CODE)
    config = transformers.AutoConfig.from_pretrained(falcon40b, trust_remote_code=TRUST_REMOTE_CODE)
    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=TRUST_REMOTE_CODE)
    print(model)
    # profiler = FlopsProfiler(model)
    # profiler.start_profile()

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    sequences = pipeline(
        "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    import pdb; pdb.set_trace()
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
