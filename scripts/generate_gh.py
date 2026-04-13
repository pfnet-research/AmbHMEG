import argparse
import pickle
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from mmengine import Config
from mmengine.runner import Runner
from tqdm.auto import tqdm

from gryphgen.gh import gh_pixelwise_channel
from gryphgen.score.dump import _to_image


@dataclass
class ValidSample:
    name: str
    valid_id: int
    rank: int
    rank_in_valid: int
    score: float
    tex: Any
    node_types: Any
    edge_types: Any
    edges: Any


def eval_pretrained(runner, weight_path):
    import torch

    orig_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = patched_load

    runner.load_checkpoint(weight_path)
    import torch

    return runner


def _cfg_valid(self, targets, noise_control=False):
    num_inference_steps = 50
    height = 224
    width = 224

    f_tgt = targets[0]
    b_tgt = targets[1]
    names = [t.name for t in f_tgt]

    batch_size = len(targets)
    assert batch_size == 2, "noise merge mode with data[0] and data[1]"
    targets = targets[0] + targets[1]

    labels = [["a"] for s in targets]  # dummy labels

    uncond_labels = [["<MASK>"] for s in targets]
    input_ids = self.tokenizer(labels, max(map(len, labels)))  # [B, ] -> [B, L]
    uncond_ids = self.tokenizer(uncond_labels, input_ids.shape[1])  # [B, ] -> [B, L]

    with torch.no_grad():
        (embeddings, valid_mask), (uncond_embeddings, uncond_valid_mask) = self._encode(
            input_ids, targets, uncond_ids=uncond_ids, mode="eval"
        )

    embeddings = torch.cat([uncond_embeddings, embeddings])
    valid_mask = torch.cat([uncond_valid_mask, valid_mask])

    batch_size = len(targets) // 2  # [uncond, cond], but conda is duplicated
    seed_latent = torch.randn(
        (1, self.unet.config.in_channels, height // 8, width // 8),
    ).to(input_ids.device)
    latents = seed_latent.repeat(batch_size, 1, 1, 1)

    latents = latents * self.noise_scheduler.init_noise_sigma
    self.noise_scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(self.noise_scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.

        latent_model_input = torch.cat([latents] * 4)

        latent_model_input = self.noise_scheduler.scale_model_input(
            latent_model_input, timestep=t
        )

        # predict the noise residual
        with torch.no_grad():
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeddings,
                encoder_attention_mask=valid_mask,  # [B, L],  `True` the mask is kept, otherwise if `False` it is discarded
            ).sample

        if noise_control:
            gs = 1

            p1, p2 = 1, 1

            # noise_pred_uncond, noise_emb1, noise_emb2 = noise_pred.chunk(3)
            noise_pred_uncond, _, noise_emb1, noise_emb2 = noise_pred.chunk(4)
            diff1 = noise_emb1 - noise_pred_uncond
            diff2 = noise_emb2 - noise_pred_uncond

            diff1, diff2, mask = gh_pixelwise_channel(
                diff1, diff2, eps=1e-6, only_adjust="both", return_mask=True
            )
            noise_c = p1 * diff1 + p2 * diff2

            noise_pred = noise_pred_uncond + gs * (noise_c)

        else:
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.noise_scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

    latents = latents / self.vae.config.scaling_factor
    with torch.no_grad():
        image = self.vae.decode(latents).sample

    ## save image with name

    data = [
        dict(
            name=n,
            image=image,
            cap_1st=ft.tex,
            cap_2nd=bt.tex,
            input_1st=asdict(ft),
            input_2nd=asdict(bt),
        )
        for (n, image, ft, bt) in zip(names, image.detach().cpu(), f_tgt, b_tgt)
    ]
    return data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/graph_sd.py",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="gryphgen_graph_sd_mathwriting_epoch_50.pth",
        help="path to weight file",
    )
    parser.add_argument(
        "--edit",
        type=str,
        default="symbol",
        choices=["symbol", "layout"],
        help="formula modification type",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="dataset split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="~/gen",
        help="path to output directory",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="~/data/mathwriting/pickles/pairs/mathwriting_symlg_graph_pairs.pkl",
        help="path to paired formula pickle file",
    )

    return parser.parse_args()


args = parse_args()
config = Config.fromfile(args.config)

pipeline = [
    dict(
        type="Annotate",
        keys=["1", "2"],
        meta=["sample_id"],
    ),
]

config.test_dataloader = dict(
    batch_size=160,
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="FormulaEditDataset",  # from gryph
        ann_file=args.source,
        pipeline=pipeline,
        filter_cfg=dict(split=args.split, edit=args.edit),
        test_mode=True,
    ),
)

save_dir = Path(args.output).joinpath(args.edit, args.split).expanduser()
save_dir.mkdir(parents=True, exist_ok=True)

runner = Runner.from_cfg(config)
dataloader = runner.test_dataloader
runner = eval_pretrained(runner, weight_path=args.weight)
model = runner.model

model._old_test_step = model.test_step
model.test_step = types.MethodType(_cfg_valid, model)
model.eval()


def _pack_manual(data):
    base_list = []
    aug_list = []
    for d in data:
        base = d.get("1")
        aug = d.get("2")
        name = d.sample_id
        base_data = ValidSample(
            name=name,
            valid_id=0,
            rank=0,
            rank_in_valid=0,
            score=0.0,
            tex=base["tex"],
            node_types=base["graph"]["node_types"],
            edge_types=base["graph"]["edge_types"],
            edges=base["graph"]["edges"],
        )

        aug_data = ValidSample(
            name=name,
            valid_id=1,
            rank=1,
            rank_in_valid=1,
            score=-1.0,
            tex=aug["tex"],
            node_types=aug["graph"]["node_types"],
            edge_types=aug["graph"]["edge_types"],
            edges=aug["graph"]["edges"],
        )

        base_list.append(base_data)
        aug_list.append(aug_data)

    return base_list, aug_list


with torch.no_grad():
    for data_batch in tqdm(dataloader, desc=args.edit):
        data = data_batch["targets"]
        batch = _pack_manual(data)

        out = model.test_step(batch, noise_control=True)
        assert len(out) == len(batch[0])

        # output
        for rec in out:
            name = rec["name"]

            image = _to_image(rec.pop("image"))

            pkl_path = save_dir / f"{name}.pkl"
            pkl_path.write_bytes(pickle.dumps(rec))

            img_path = save_dir / f"{name}.png"
            image.save(img_path)
