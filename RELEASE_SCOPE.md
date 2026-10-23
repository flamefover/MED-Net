# Release scope

Release: v0.1


This release provides the MED-Net architecture configuration and the module implementations listed below.

The architecture configuration and module interfaces are included. The following list specifies implementation availability in this release.

## mednet/frj.py

Provided implementations: `check_requirements`, `select_codes`.

Scheduled implementations: `sample_gpu_occupancy`.

## mednet/model.py

Provided implementations: `Concat.__init__`, `Concat.forward`, `MEDNet.__init__`.

Scheduled implementations: `MEDNet._apply`, `MEDNet.forward`.

## mednet/modules/amdf.py

Provided implementations: `AMDFDownsample.__init__`, `AMDFDownsample.forward`, `AMDFFusion.__init__`, `AMDFFusion.forward`, `C3.__init__`, `C3.forward`, `C3k_EMBC.__init__`, `Conv.forward_fuse`, `EffectiveSEModule.__init__`, `EffectiveSEModule.forward`, `MBConv.__init__`, `MBConv.forward`.

Scheduled implementations: `Bottleneck.__init__`, `Bottleneck.forward`, `C3k.__init__`, `Conv.__init__`, `Conv.forward`, `autopad`.

## mednet/modules/common.py

Provided implementations: `Attention.__init__`, `Attention.forward`, `C2PSA.__init__`, `C2PSA.forward`, `Conv.__init__`, `Conv.forward`, `Conv.forward_fuse`, `PSABlock.__init__`, `PSABlock.forward`, `SPPF.__init__`, `SPPF.forward`, `autopad`.

Scheduled implementations: None.

## mednet/modules/msc_repvit.py

Provided implementations: `MFG.__init__`, `MFG.forward`, `Conv2d_BN.__init__`, `MSCRepViT.__init__`, `MSCRepViT.forward`, `MSCRepViT.switch_to_deploy`, `Mix.__init__`, `Mix.forward`, `RepVGGDW.__init__`, `RepVGGDW.forward`, `RepVGGDW.fuse_self`, `RepViT.__init__`, `RepViT.forward`, `RepViT.switch_to_deploy`, `RepViTBlock.__init__`, `RepViTBlock.forward`, `Residual.__init__`, `Residual.forward`, `Residual.fuse_self`, `SqueezeExcite.__init__`, `SqueezeExcite.forward`, `_make_divisible`.

Scheduled implementations: `Conv2d_BN.fuse_self`, `replace_batchnorm`.

## mednet/modules/udeh.py

Provided implementations: `Conv2d_hd.__init__`, `Conv2d_vd.__init__`, `Conv2d_vd.get_weight`, `Conv_GN.forward`, `DEConv.__init__`, `DEConv.forward`, `DEConv.switch_to_deploy`, `DEConv_GN.__init__`, `UDEH.__init__`, `UDEH.bias_init`, `UDEH.decode_bboxes`, `UDEH.forward`, `autopad`.

Scheduled implementations: `Conv2d_cd.__init__`, `Conv2d_cd.get_weight`, `Conv2d_hd.get_weight`, `Conv_GN.__init__`, `DFL.__init__`, `DFL.forward`, `Scale.__init__`, `Scale.forward`, `dist2bbox`, `make_anchors`.

## Runtime and evidence

Scheduled methods explicitly raise NotImplementedError when called. No identity substitutions or fabricated outputs are used. An available method may still depend on a scheduled method. Installation/import alone does not imply that end-to-end execution is available.

Reference training, evaluation, split-manifest validation and device profiling tools are provided in v1.0. Datasets and trained weights are not bundled; supply a labeled dataset to train and evaluate the network.
