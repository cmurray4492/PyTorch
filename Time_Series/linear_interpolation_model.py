import torch
import numpy as np

from scipy import interpolate


class INterpolationPredictor(torch.nn.Module):
    def forward(self, x):
        last_values = []
        values = x.tolist()
        for v in values:
            x = np.arange(0, len(v))
            y = interpolate.interp1d(x, v, fill_value='extrapolate')
            last_values.append([y(len(v)).tolist()])
        return torch.tensor(data=last_values)
