import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by the number of heads"

        # Linear layers for Q, K, V projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Output linear layer
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        # Compute Q, K, V
        Q = self.query(x)  # Shape: (batch_size, seq_length, embed_dim)
        K = self.key(x)    # Shape: (batch_size, seq_length, embed_dim)
        V = self.value(x)  # Shape: (batch_size, seq_length, embed_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads,
                   self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads,
                   self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads,
                   self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # Shape: (batch_size, num_heads, seq_length, seq_length)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute weighted sum of values
        # Shape: (batch_size, num_heads, seq_length, head_dim)
        weighted_sum = torch.matmul(attention_weights, V)

        # Concatenate heads and pass through final linear layer
        weighted_sum = weighted_sum.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.embed_dim)
        # Shape: (batch_size, seq_length, embed_dim)
        output = self.fc_out(weighted_sum)

        return output


# Example usage
embed_dim = 8  # Embedding size
num_heads = 2  # Number of attention heads
seq_length = 4  # Length of the input sequence

# Initialize random input (batch_size=1, seq_length=4, embed_dim=8)
x = torch.rand(1, seq_length, embed_dim)

# Initialize and apply the self-attention module
self_attention = SelfAttention(embed_dim, num_heads)
output = self_attention(x)

print("Input Shape:", x.shape)
print("Output Shape:", output.shape)
