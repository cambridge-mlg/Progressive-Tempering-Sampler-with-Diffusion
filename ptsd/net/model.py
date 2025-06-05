import torch.nn as nn
import torch
from ptsd.net.egnn import EGNN_dynamics, EGNN_atom


class TimeEmbedding(nn.Module):

    def __init__(self, t_embed_dim, scale=30.0):
        super().__init__()

        self.register_buffer("w", torch.randn(t_embed_dim // 2) * scale)

    def forward(self, t):
        # t: (B, )
        t_proj = 2.0 * torch.pi * self.w[None, :] * t[:, None]  # (B, E//2)
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)  # (B, E)
        return t_embed


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        t_embed_dim,
        n_layers=3,
        act_fn=torch.nn.SiLU(),
        skip_connect=True,
        data_sigma=0.5,
    ):
        super().__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * n_layers
        assert len(hidden_dims) == n_layers
        net = []
        net.append(nn.Linear(in_dim + t_embed_dim, hidden_dims[0]))
        for idx in range(len(hidden_dims) - 1):
            net.append(nn.Linear(hidden_dims[idx] + t_embed_dim, hidden_dims[idx + 1]))
        net.append(nn.Linear(hidden_dims[-1] + t_embed_dim, in_dim))

        self.t_emb = PositionalEmbedding(
            t_embed_dim
        )  # TimeEmbedding(t_embed_dim, scale=30.0)

        self.net = nn.ModuleList(net)
        self.act = act_fn

        # Count function calls
        self.counter = 0

        self.skip_connect = skip_connect
        self.data_sigma = data_sigma

    def forward(self, logt, xs):
        self.counter += 1
        data_sigma = self.data_sigma

        t_emb = self.t_emb(logt / 4)

        x_in = xs.clone()

        t = logt.exp()[:, None]
        c_in = 1 / (data_sigma**2 + t**2) ** 0.5

        xs = torch.cat([xs * c_in, t_emb], -1)
        for idx, layer in enumerate(self.net):
            xs = layer(xs)
            if idx != len(self.net) - 1:
                xs = self.act(xs)
                xs = torch.cat([xs, t_emb], -1)
        if self.skip_connect:
            out = (
                x_in * data_sigma**2 / (data_sigma**2 + t**2)
                + xs * data_sigma * t / (data_sigma**2 + t**2) ** 0.5
            )
            return out
        else:
            return xs


class EGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        n_particles,
        n_dimensions,
        hidden_dims,
        t_embed_dims,
        n_layers=3,
        act_fn=torch.nn.SiLU(),
        skip_connect=True,
        data_sigma=0.5,
        device='cuda',
    ):
        super().__init__()
        self.device = device
        assert in_dim == n_particles * n_dimensions
        self.net = EGNN_dynamics(
            n_particles=n_particles,
            n_dimension=n_dimensions,
            hidden_nf=hidden_dims,
            t_emb_dims=t_embed_dims,
            n_layers=n_layers,
            attention=True,
            condition_time=False,
            tanh=True,
            act_fn=act_fn,
            device=device,
        )
        self.t_emb = PositionalEmbedding(
            t_embed_dims
        )  # TimeEmbedding(t_embed_dim, scale=30.0)

        self.counter = 0

        self.skip_connect = skip_connect
        self.data_sigma = data_sigma

    def forward(self, logt, xs):
        self.counter += 1
        data_sigma = self.data_sigma

        t_emb = self.t_emb(logt / 4)

        x_in = xs.clone()

        t = logt.exp()[:, None]
        c_in = 1 / (data_sigma**2 + t**2) ** 0.5

        xs = self.net(t_emb, xs * c_in)
        if self.skip_connect:
            out = (
                x_in * data_sigma**2 / (data_sigma**2 + t**2)
                + xs * data_sigma * t / (data_sigma**2 + t**2) ** 0.5
            )
            return out
        else:
            return xs


class EGNN_ALDP(nn.Module):
    def __init__(
        self,
        in_dim,
        n_particles,
        n_dimensions,
        hidden_dims,
        t_embed_dims,
        atom_type_embedding_dim,
        n_layers=3,
        act_fn=torch.nn.SiLU(),
        skip_connect=True,
        data_sigma=0.5,
        atom_type_class="element",
    ):
        super().__init__()
        assert in_dim == n_particles * n_dimensions
        self.bonds = [
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (4, 5),
            (4, 6),
            (6, 7),
            (6, 8),
            (8, 9),
            (8, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (8, 14),
            (14, 15),
            (14, 16),
            (16, 17),
            (16, 18),
            (18, 19),
            (18, 20),
            (18, 21),
        ]
        if atom_type_class == "element":
            self.atom_type_labels = [
                0,  # H 0
                1,  # C 1
                0,  # H 2
                0,  # H 3
                1,  # C 4
                2,  # O 5
                3,  # N 6
                0,  # H 7
                1,  # C 8
                0,  # H 9
                1,  # C 10
                0,  # H 11
                0,  # H 12
                0,  # H 13
                1,  # C 14
                2,  # O 15
                3,  # N 16
                0,  # H 17
                1,  # C 18
                0,  # H 19
                0,  # H 20
                0,  # H 21
            ]
        elif atom_type_class == "structure":
            self.atom_type_labels = [
                0,  # H1 0
                1,  # CH3 1
                2,  # H2 2
                3,  # H3 3
                4,  # C 4
                5,  # O 5
                6,  # N 6
                7,  # H 7
                8,  # CA 8
                9,  # HA 9
                10,  # CB 10
                11,  # HB1 11
                12,  # HB2 12
                13,  # HB3 13
                4,  # C 14
                5,  # O 15
                6,  # N 16
                7,  # H 17
                4,  # C 18
                0,  # H1 19
                2,  # H2 20
                3,  # H3 21
            ]
        else:
            raise ValueError("atom_type_class must be either 'element' or 'structure'")
        self.net = EGNN_atom(
            n_particles=n_particles,
            n_dimension=n_dimensions,
            atom_type_labels=self.atom_type_labels,
            bonds=self.bonds,
            hidden_nf=hidden_dims,
            device='cuda',
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=False,
            attention=True,
            condition_time=True,
            tanh=False,
            agg="sum",
            time_embedding_dim=t_embed_dims,
            atom_type_embedding_dim=atom_type_embedding_dim,
        )

        self.t_emb = PositionalEmbedding(
            t_embed_dims
        )  # TimeEmbedding(t_embed_dim, scale=30.0)

        self.counter = 0

        self.skip_connect = skip_connect
        self.data_sigma = data_sigma

    def forward(self, logt, xs):
        self.counter += 1
        data_sigma = self.data_sigma

        t_emb = self.t_emb(logt / 4)

        x_in = xs.clone()

        t = logt.exp()[:, None]
        c_in = 1 / (data_sigma**2 + t**2) ** 0.5

        xs = self.net(t_emb, xs * c_in)
        if self.skip_connect:
            out = (
                x_in * data_sigma**2 / (data_sigma**2 + t**2)
                + xs * data_sigma * t / (data_sigma**2 + t**2) ** 0.5
            )
            return out
        else:
            return xs
