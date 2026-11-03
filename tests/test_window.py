import torch

from motorphen.inference import predict_last, reconstruct
from motorphen.model import GaitMAE


def test_longer_window_inference_uses_model_window():
    torch.manual_seed(0)
    model = GaitMAE(n_frames=15, enc_layers=1, dec_layers=1).eval()
    seq = torch.rand(40, 12, 3) - 0.5
    assert predict_last(model, seq).shape == (26, 12, 3)
    out = reconstruct(model, seq, [8])
    assert torch.equal(out[:14], seq[:14]) and out.shape == seq.shape
