import numpy as np

SIGMOID_FUNC = lambda x: 1 / (1 + np.exp(-x.detach().cpu().numpy()))