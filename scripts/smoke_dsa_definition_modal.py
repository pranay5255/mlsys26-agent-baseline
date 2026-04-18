import json
import math
from pathlib import Path

import cutlass
import cutlass.cute as cute
import modal

APP_NAME = "dsa-ps1-definition-smoke"
DEFINITION_PATH = Path(
    "datasets/mlsys26-contest/definitions/dsa_paged/"
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1.json"
)


app = modal.App(APP_NAME)
image = modal.Image.from_registry(
    "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04",
    add_python="3.11",
).pip_install("torch", "nvidia-cutlass-dsl[cu13]")


@cute.kernel
def empty_cute_kernel():
    pass


@cute.jit
def launch_empty_cute_kernel(stream):
    empty_cute_kernel().launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
        stream=stream,
    )


@app.function(image=image, gpu="B200", timeout=300)
def run_definition_smoke(definition_json: str) -> dict:
    import cutlass
    import cutlass.cute as cute
    import torch

    definition = json.loads(definition_json)
    namespace = {}
    exec(definition["reference"], namespace)
    reference_run = namespace["run"]

    device = torch.device("cuda")
    num_tokens = 1
    num_pages = 2048
    topk = 2048

    torch.manual_seed(0)
    q_nope = torch.randn((num_tokens, 16, 512), dtype=torch.bfloat16, device=device)
    q_pe = torch.randn((num_tokens, 16, 64), dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn((num_pages, 1, 512), dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn((num_pages, 1, 64), dtype=torch.bfloat16, device=device)
    sparse_indices = torch.arange(topk, dtype=torch.int32, device=device).reshape(
        num_tokens, topk
    )
    sm_scale = 1.0 / math.sqrt(192.0)

    output, lse = reference_run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    torch.cuda.synchronize()

    stream = cutlass.cuda.default_stream()
    compiled = cute.compile(launch_empty_cute_kernel, stream)
    compiled(stream)
    cutlass.cuda.stream_sync(stream)

    return {
        "device": torch.cuda.get_device_name(0),
        "cute_empty_kernel_compiled": True,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "lse_shape": list(lse.shape),
        "lse_dtype": str(lse.dtype),
        "output_finite": bool(torch.isfinite(output.float()).all().item()),
        "lse_finite": bool(torch.isfinite(lse).all().item()),
        "lse_mean": float(lse.mean().item()),
    }


def run_smoke() -> dict:
    definition_json = DEFINITION_PATH.read_text()
    modal.enable_output()
    with app.run():
        result = run_definition_smoke.remote(definition_json)
    print(result)
    return result


def main() -> None:
    run_smoke()


if __name__ == "__main__":
    main()
