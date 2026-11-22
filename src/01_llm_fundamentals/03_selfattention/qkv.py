import numpy as np

X = np.array([
    [1, 0, 1],
    [0, 2, 1],
    [0, 1, 1],
])

np.random.seed(42)

# this values get computed while training but for now we are taking random values for understanding
Wq = np.random.rand(3, 3)
Wk = np.random.rand(3, 3)
Wv = np.random.rand(3, 3)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

def self_attention(Q,K,V):
    d_k= K.shape[1]
    # casual masking to mask future tokens while processsing current token
    score= (Q @ K.T) / np.sqrt(d_k)
    
    seq_len=score.shape[0]
    mask = np.triu(
        np.ones((seq_len, seq_len)),
        k=1
    )

    score = np.where(mask == 1, -np.inf, score)
    # soft max
    attention_weights= np.exp(score)/np.sum(
        np.exp(score),axis=1,keepdims=True
    )

    output=attention_weights @ V
    return output,attention_weights


output,attention_weights=self_attention(Q,K,V)
print(output,attention_weights)