# MED-Net implementation notes

This document describes the MED-Net architecture, configuration and execution interfaces.

## Implemented correspondence

| Network component | Implementation |
| --- | --- |
| MSC-RepViT stages 3/6/9/3 | `MSCRepViT.depths` configures the number of blocks in each stage |
| MFG on each backbone feature | Four independent gates at strides 4/8/16/32 |
| Stage I weighted downsampling | `AMDFDownsample` |
| Stage II path choice | `AMDFFusion`, code 0 Efficient/Wide, code 1 Inverted/Deep |
| Wise-Joint element-wise aggregation | Equal-channel direct and processed streams are added, then projected |
| FRJ before detection | `select_codes` implements below 70%, 70%–85%, and at least 85% cases; forward never samples resources |
| UDEH Conv Series | Depthwise 3×3, pointwise projection, GN/SiLU, shared detail enhancement |
| Four detail branches | Standard, central, horizontal and vertical |
| Scale and IFB | Per-level regression Scale and multi-level distribution-based box decoding retained |

## Run and inspect

The full-model example below requires v1.0. Earlier releases provide a runnable
backbone example in README.md and explicitly gate unavailable methods.

```sh
pip install -e .
python -B verify_release.py
```

```python
import torch
from mednet import MEDNet
from mednet.frj import sample_gpu_occupancy

# Measure once BEFORE model allocation. This is global device memory occupancy.
occupancy = sample_gpu_occupancy(device=0)
model = MEDNet(nc=4, occupancy=occupancy).cuda().eval()
print(model.route_codes)
with torch.no_grad():
    predictions, raw = model(torch.zeros(1, 3, 640, 640, device='cuda'))
```

CPU architecture tests can explicitly pass a simulated ratio, e.g. `MEDNet(nc=4, occupancy=0.75)`. This is a test input, not a resource measurement. `route_mode='wide'` or `'deep'` disables selection and fixes all paths for ablation. Reconstruct the model for another route; do not change routing after loading a route-specific checkpoint.

## Adopted implementation choices

- FRJ uses the fixed forward-execution order `[10,13,16,19,22,25]`. Routing is selected at initialization using this priority order.
- Wise-Joint adds the direct stream to the final output of the selected path, followed by a pointwise output projection. Both streams have matching channels and spatial dimensions. The direct stream bypasses path processing; no intermediate states are added to Wise-Joint. This implements the described end-of-path element-wise aggregation without changing spatial resolution.
- The Efficient Module retains its existing internal C3k-style concatenation. This internal operation is distinct from the outer Wise-Joint addition.
- The default width/depth scale is `n`; the model configuration includes SPPF and C2PSA.
- MFG uses independent parameters at each scale because channel counts differ.
- `check_requirements` is an explicit interface for a measured accuracy/latency/memory profile. No validated profiles are supplied, and model construction does not pretend these requirements have passed. Before deployment, the caller must check the exact selected route and checkpoint on the target device and reject a failing profile.

## Scope

Model implementations are provided in stages, with regression tests for available components in every release. v1.0 adds a reference training/evaluation workflow and tools to validate exact user-supplied splits and measure route resources; see EXPERIMENTS.md in v1.0. Datasets and trained weights must be supplied separately. See RELEASE_SCOPE.md for availability.



## Correctness and deployment

- Backbone stage outputs are selected from graph boundaries, without a random
  initialization probe or BatchNorm updates. The full model requires nonempty
  Bx3xHxW input with positive H/W divisible by 32; the backbone also accepts odd sizes.
- Deployment conversion requires eval mode. Fused convolutions preserve device
  and dtype, and residual identity kernels preserve channels, including grouped
  and depthwise cases. Retain the unfused checkpoint and check numerical equivalence.
- Difference-convolution helpers explicitly support kernel_size=3. Other sizes
  raise ValueError at construction.
- FRJ thresholds are a fixed initialization heuristic. Path names and parameter
  counts do not prove resource savings. Use route-specific trained checkpoints
  and target-device measurements to check requirements before deployment.
