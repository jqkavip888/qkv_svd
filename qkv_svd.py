import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
import zhplot

# set some hyperparameters
vocab_size = 1000
batch_size = 16
seq_length = 20
embedding_dim = 768
d_k_list = [16,32,64,128]

class SingleHeadAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, d_k):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.W_Q = nn.Parameter(torch.randn(embedding_dim, d_k))
        self.W_K = nn.Parameter(torch.randn(embedding_dim, d_k))
        self.W_V = nn.Parameter(torch.randn(embedding_dim, d_k))
        self.LM_head = nn.Linear(d_k, vocab_size)
        self.d_k = d_k

    def forward(self, input_token):
        x = self.embedding(input_token)
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        scores = Q @ K.transpose(-2, -1)  / (self.d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        output = weights @ V
        x_int = self.LM_head(output)

        return x_int


def train():
    for d_k in d_k_list:
        torch.manual_seed(42)
        model = SingleHeadAttention(vocab_size, embedding_dim, d_k)
        optimizer = AdamW(model.parameters(), lr=3e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(2000):
            input_token = torch.randint(0, vocab_size, (batch_size, seq_length))
            output = model(input_token)
            loss = criterion(output.transpose(1,2), input_token)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss {loss.item():.3f}')

        print(f'd_k={d_k} 训练完成')

        W_Q = model.W_Q.detach()
        W_K = model.W_K.detach()
        QK = W_Q @ W_K.T
        _, S, _ = torch.linalg.svd(QK)
        print(S)        # Singular value

        plt.plot(S.numpy(), label=f'd_k={d_k}')
        plt.axvline(x=d_k, linestyle=':', alpha=0.5)


def plot_svd():
    plt.yscale('log')
    plt.xlabel('奇异值索引 Singular value index')
    plt.ylabel('奇异值大小（log scale）')
    plt.title('W_Q @ W_K.T 奇异值谱 Singular value Spectrum')
    plt.axvline(x=16, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=32, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=64, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=128, color='gray', linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
    plot_svd()



