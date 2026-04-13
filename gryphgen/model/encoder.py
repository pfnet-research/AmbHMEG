"""
Copy and paste from BBoxNet, https://github.com/AiArt-Gao/HMEG/blob/main/model/bbox_net.py
"""

import pickle
from pathlib import Path

import positional_encodings.torch_encodings as P
import torch
import torch.nn as nn

from gryphgen.hmeg.blocks import AttentionModule, weight_init
from gryphgen.hmeg.graph import GraphTripleConv, GraphTripleConvNet
from gryphgen.hmeg.layer import build_mlp
from gryphgen.utils import MODELS


@MODELS.register_module()
class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
        pad_id: int,
        mask_id: int,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = P.PositionalEncoding1D(channels=d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,  # [B, L, D]
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.pad_id = pad_id
        self.mask_id = mask_id

    def forward(self, input_ids):
        # input_ids: [B, L]
        B, L = input_ids.shape

        tok = self.token_emb(input_ids)  # [B, L, D]

        pos = self.pos_emb(tok)  # [B, L, D]
        x = tok + pos  # [B, L, D]

        # ignore pad token
        src_key_padding_mask = input_ids == self.pad_id

        # warning
        all_pad = src_key_padding_mask.all(dim=1)
        if all_pad.any():
            print("[WARN]", "all tokens are pad token")

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, L, D]
        x = self.ln(x)
        return x  # [B, L, D]


@MODELS.register_module()
class GraphEncoder(nn.Module):
    def __init__(
        self,
        vocab_load,
        embedding_dim=128,
        gconv_dim=128,
        gconv_hidden_dim=512,
        gconv_pooling="avg",
        gconv_num_layers=5,
        mlp_normalization="none",
    ):
        super().__init__()

        vocab_load = Path(vocab_load).expanduser()
        with open(vocab_load, "rb") as f:
            vocab = pickle.load(f)

        self.vocab = vocab
        num_objs = len(vocab["node"])
        num_preds = len(vocab["edge"])
        self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

        if gconv_num_layers == 0:
            self.gconv = nn.Linear(embedding_dim, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                "input_dim": embedding_dim,
                "output_dim": gconv_dim,
                "hidden_dim": gconv_hidden_dim,
                "pooling": gconv_pooling,
                "mlp_normalization": mlp_normalization,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                "input_dim": gconv_dim,
                "hidden_dim": gconv_hidden_dim,
                "pooling": gconv_pooling,
                "num_layers": gconv_num_layers - 1,
                "mlp_normalization": mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)
        box_net_dim = 4
        box_net_layers = [gconv_dim * 2 + 64, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)
        self.graph_encoder = AttentionModule(gconv_dim)
        self.apply(weight_init)

    def forward(
        self,
        objs,
        triples,
        # noise,
        obj_to_img=None,
    ):
        O, _ = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)
        graph_vecs = self.graph_encoder(obj_vecs, obj_to_img)

        vecs = torch.cat([obj_vecs, graph_vecs], dim=1)

        return vecs
