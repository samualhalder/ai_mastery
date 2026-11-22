import numpy as np
from multi_head_attention import MultiHeadAttention

np.random.seed(42)

x=np.random.rand(5,8)

mha= MultiHeadAttention(
    d_model=8,
    h_count=2
)

output=mha.forward(x)

print(output)