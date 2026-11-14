# MED-Net

Release v0.2.

## Version updates

The code is being released incrementally. We will continue to improve the implementation and documentation through subsequent releases.

- [v0.2](https://github.com/flamefover/MED-Net/releases/tag/v0.2) (14 November 2026): additional UDEH implementations and updated release documentation.
- [v0.1](https://github.com/flamefover/MED-Net/releases/tag/v0.1): the initial source snapshot uploaded on 23 October 2026.

See the [changelog](CHANGELOG.md), [release history](https://github.com/flamefover/MED-Net/releases), and [v0.1-to-v0.2 code comparison](https://github.com/flamefover/MED-Net/compare/v0.1...v0.2) for the changes between versions.


MED-Net is an object-detection model combining an MSC-RepViT backbone, per-scale
MFG attention, AMDF feature fusion, initialization-time FRJ route selection and
a UDEH detection head.

This release includes an independently runnable MSC-RepViT backbone and the UDEH components described in the changelog.

## Install and verify

```sh
pip install -e .
python -B verify_release.py
```

## Example

```python
import torch
from mednet.modules.msc_repvit import MSCRepViT

backbone = MSCRepViT().eval()
with torch.no_grad():
    features = backbone(torch.zeros(1, 3, 640, 640))
```

The complete model graph and reference experiment workflow are planned for v1.0. See [RELEASE_SCOPE.md](RELEASE_SCOPE.md)
for this version's exact interfaces and availability.


See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for architecture, configuration and input constraints.
