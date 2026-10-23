import copy
import json
from pathlib import Path
import unittest

import torch
from torch import nn
from mednet.modules.msc_repvit import MSCRepViT, Residual, Conv2d_BN
from mednet.modules.udeh import Conv2d_vd, Conv2d_cd, DEConv_GN, UDEH

ROOT = Path(__file__).resolve().parents[1]
GATES = {x['symbol'] for x in json.loads((ROOT/'release_manifest.json').read_text())['scheduled']}


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_constructor_preserves_batchnorm(self):
        m = MSCRepViT()
        bns = [x for x in m.modules() if isinstance(x, nn.BatchNorm2d)]
        self.assertTrue(bns)
        for bn in bns:
            self.assertEqual(int(bn.num_batches_tracked), 0)
            torch.testing.assert_close(bn.running_mean, torch.zeros_like(bn.running_mean))
            torch.testing.assert_close(bn.running_var, torch.ones_like(bn.running_var))

    def test_stage_outputs_for_odd_rectangular_image(self):
        m = MSCRepViT().eval()
        with torch.no_grad():
            ys = m(torch.randn(1, 3, 65, 97))
        self.assertEqual([tuple(y.shape) for y in ys], [(1,48,17,25),(1,96,9,13),(1,192,5,7),(1,384,3,4)])

    def test_residual_identity_dense_grouped_depthwise(self):
        for groups in (1, 2, 4):
            for kernel in (1, 3, 5):
                r = Residual(nn.Conv2d(4, 4, kernel, padding=kernel//2, groups=groups)).eval().double()
                x = torch.randn(2, 4, 9, 9, dtype=torch.float64)
                before_state = copy.deepcopy(r.state_dict())
                with torch.no_grad():
                    reference = r(x)
                    fused = r.fuse_self()
                    torch.testing.assert_close(reference, fused(x), rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(fused(x), r.fuse_self()(x))
                for key, value in r.state_dict().items():
                    torch.testing.assert_close(value, before_state[key])

    def test_training_fusion_rejected(self):
        with self.assertRaisesRegex(ValueError, 'eval'):
            Residual(nn.Conv2d(4, 4, 3, padding=1)).fuse_self()

    @unittest.skipIf('Conv2d_BN.fuse_self' in GATES, 'Scheduled in this stage')
    def test_fusion_preserves_dtype_and_values(self):
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            block = Conv2d_BN(4, 4, 3, 1, 1).eval().to(dtype=dtype)
            fused = block.fuse_self()
            self.assertEqual(fused.weight.dtype, dtype)
            self.assertEqual(fused.bias.dtype, dtype)
            self.assertFalse(fused.training)
            if dtype in (torch.float32, torch.float64):
                x = torch.randn(2,4,8,8,dtype=dtype)
                with torch.no_grad():
                    torch.testing.assert_close(block(x), fused(x))

    @unittest.skipIf('replace_batchnorm' in GATES, 'Scheduled in this stage')
    def test_backbone_double_deploy(self):
        m = MSCRepViT().eval().double()
        x = torch.randn(1,3,64,64,dtype=torch.float64)
        with torch.no_grad():
            before = m(x)
            m.switch_to_deploy()
            after = m(x)
            for a,b in zip(before, after):
                torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)

    def test_difference_kernel_validation(self):
        with self.assertRaisesRegex(ValueError, 'kernel_size=3'):
            Conv2d_vd(4,4,kernel_size=5)

    @unittest.skipIf('Conv2d_cd.__init__' in GATES, 'Scheduled in this stage')
    def test_central_difference_theta(self):
        m = Conv2d_cd(4,4,theta=0)
        w, _ = m.get_weight()
        torch.testing.assert_close(w,m.conv.weight)
        m.theta = 1
        w, _ = m.get_weight()
        torch.testing.assert_close(w.sum((-1,-2)), torch.zeros(4,4), atol=1e-6, rtol=0)

    @unittest.skipIf('Conv2d_cd.__init__' in GATES, 'Scheduled in this stage')
    def test_detail_deploy_is_repeatable(self):
        m = DEConv_GN(16).eval().double()
        x = torch.randn(2,16,8,8,dtype=torch.float64)
        with torch.no_grad():
            expected = m(x)
            m.switch_to_deploy()
            m.switch_to_deploy()
            torch.testing.assert_close(expected,m(x))

    @unittest.skipIf('Conv_GN.__init__' in GATES, 'Scheduled in this stage')
    def test_head_buffers_convert_with_model(self):
        head = UDEH(nc=4,hidc=64,ch=(64,128,256)).double()
        for name in ('stride','anchors','strides'):
            self.assertEqual(getattr(head,name).dtype,torch.float64)


if __name__ == '__main__':
    unittest.main()
