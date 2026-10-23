# MED-Net

Release v0.1.

MED-Net is an object-detection model combining an MSC-RepViT backbone, per-scale
MFG attention, AMDF feature fusion, initialization-time FRJ route selection and
a UDEH detection head.

This is a staged model-source release. The backbone can run independently; full MEDNet execution is not available because some dependencies are scheduled for later release.

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

Scheduled methods raise `NotImplementedError`. Use v1.0 for the complete model
graph and reference experiment workflow. See [RELEASE_SCOPE.md](RELEASE_SCOPE.md)
for this version's exact interfaces and availability.


See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for architecture, configuration and input constraints.
