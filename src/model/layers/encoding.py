import torch.nn as nn
import numpy as np
import torch
torch.cuda.empty_cache()


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)


class DisentangleTimeEncode(torch.nn.Module):
    def __init__(self, components, expand_dim, factor=5):
        super(DisentangleTimeEncode, self).__init__()

        time_dim = expand_dim
        self.factor = factor
        self.components = components

        init_freq = np.zeros((self.components, time_dim))
        for i in range(self.components):
            # span_range = 2 + int(np.exp(i))
            span_range = 10 + i
            init_freq[i, :] = 1 / \
                span_range ** np.linspace(0, (span_range - 1), time_dim)
        self.basis_freq = torch.nn.Parameter(
            torch.from_numpy(init_freq).float())
        self.phase = torch.nn.Parameter(
            torch.zeros(self.components, time_dim).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1, 1)  # [N, L, 1, 1]
        # [N, L, components, time_dim]
        map_ts = ts * self.basis_freq.view(1, self.components, -1)
        map_ts += self.phase.view(1, self.components, -1)

        harmonic = torch.cos(map_ts)

        return harmonic


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(
            num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out
