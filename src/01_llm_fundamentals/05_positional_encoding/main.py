import numpy as np
from positional_encoding import positional_encoding

X = np.random.randn(5, 8)

PE = positional_encoding(
    seq_len=5,
    d_model=8
)

X_with_position = X + PE
PE = positional_encoding(
    seq_len=5,
    d_model=8
)

np.set_printoptions(precision=4, suppress=True)

print(PE)
print("Original:", X.shape)
print("Positional encoding:", PE.shape)
print("Final:", X_with_position.shape)