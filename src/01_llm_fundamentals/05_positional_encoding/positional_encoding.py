import numpy as np

def positional_encoding(seq_len, d_model):

    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):

            denominator = 10000 ** (i / d_model)

            PE[pos, i] = np.sin(pos / denominator)

            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / denominator)

    return PE