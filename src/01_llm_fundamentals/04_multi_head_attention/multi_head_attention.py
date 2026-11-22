import numpy as np

class MultiHeadAttention:

    def __init__(self, d_model, h_count):
        assert d_model%h_count==0

        self.d_model=d_model
        self.num_heads=h_count
        self.d_head= self.d_model//self.num_heads
        
        self.Wq = np.random.randn(
             d_model, d_model
        )

        self.Wk = np.random.randn(
            d_model, d_model
        )

        self.Wv = np.random.randn(
            d_model, d_model
        )

        self.Wo = np.random.randn(
            d_model, d_model
        )

    def split_head(self,x):
        seq_len=x.shape[0]

        x=x.reshape(
            seq_len,
            self.num_heads,
            self.d_head
        )

        return x
    
    def scaled_dot_product_attention(self, Q, K, V):

        scores = (Q @ K.T) / np.sqrt(self.d_head)

        # Causal mask
        seq_len = scores.shape[0]
        mask = np.triu(
            np.ones((seq_len, seq_len)),
            k=1
        )

        scores = np.where(mask == 1, -np.inf, scores)

        # Softmax
        exp_scores = np.exp(scores)
        attention_weights = exp_scores / np.sum(
            exp_scores,
            axis=1,
            keepdims=True
        )

        output = attention_weights @ V

        return output, attention_weights

    def forward(self,x):

        Q= x @ self.Wq
        K= x @ self.Wk
        V= x @ self.Wv

        Q= self.split_head(Q)
        K= self.split_head(K)
        V= self.split_head(V)

        head_outputs=[]

        for i in range(self.num_heads):
            q=Q[:,i,:]
            k=K[:,i,:]
            v=V[:,i,:]

            output,w=self.scaled_dot_product_attention(q,k,v)
            head_outputs.append(output)

        concatenated=np.concatenate(head_outputs,axis=1)

        output= concatenated @ self.Wo
        return output        