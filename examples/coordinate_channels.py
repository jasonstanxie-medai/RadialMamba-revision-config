#!/usr/bin/env python3
"""CPU example using the supplied coordinate module, without a dataset."""
from pathlib import Path
import sys
try:
    import torch
except ModuleNotFoundError:
    raise SystemExit('This example requires PyTorch. Run it in your model environment. '
                     'For verification without PyTorch, run scripts/verify_bundle.py.')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coord_conv import AddCoordinates

if __name__ == '__main__':
    rgb = torch.zeros(1, 3, 512, 512)
    out = AddCoordinates(with_r=True)(rgb)
    print('Input shape:', tuple(rgb.shape))
    print('Output shape:', tuple(out.shape))
    for c, name in zip((3, 4, 5), ('x (height)', 'y (width)', 'r')):
        print(name, 'range:', float(out[:, c].min()), float(out[:, c].max()))
    assert out.shape == (1, 6, 512, 512)
    assert torch.allclose(out[0, 3, :, 0], torch.linspace(-1, 1, 512))
    assert torch.allclose(out[0, 4, 0, :], torch.linspace(-1, 1, 512))
    assert torch.allclose(out[:, 5], torch.sqrt(out[:, 3]**2 + out[:, 4]**2)/(2**0.5))
    print('Coordinate convention verified.')
