import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from mmengine.model import BaseModel

from gryphgen.hmeg import graph_preprocesser
from gryphgen.utils import MODELS, build

from .utils import pad_embeddings


@MODELS.register_module()
class StableDiffusion(BaseModel):
    def __init__(
        self,
        model_id: str,
        vocab: dict,
        text_encoder: dict | None,
        p_uncond: float,
        guidance_scale: float,
        graph_encoder=None,
    ):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_config(model_id, subfolder="unet")
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.tokenizer = build(vocab)

        self.encoder = None
        self.encoder_type = None
        if text_encoder:
            self.encoder = build(
                text_encoder,
                pad_id=self.tokenizer.lexicon.SKIP,
                mask_id=self.tokenizer.lexicon.MASK,
                vocab_size=self.tokenizer.lexicon.num_class,
            )
            self.encoder_type = text_encoder["type"]
        if graph_encoder:
            self.encoder = build(graph_encoder)
            self.encoder_type = graph_encoder["type"]
            self.graph_null_emb = nn.Parameter(
                torch.zeros(int(graph_encoder.embedding_dim + graph_encoder.gconv_dim))
            )

        self.vae.requires_grad_(False)
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale

    def init_weights(self):
        pass

    def forward(self, mode: str, **kwargs):
        if mode == "loss":
            return self._train(**kwargs)
        else:
            return self._valid(**kwargs)

    def _encode(self, input_ids, targets, uncond_ids=None, mode="eval"):
        B = len(targets)

        embeddings = None
        uncond_embeddings = None
        valid_mask = None
        uncond_valid_mask = None

        if self.encoder_type == "TextEncoder":
            if mode == "train":
                # when training, input_ids is replaced randomly.
                cond = torch.rand(len(input_ids), 1, device=input_ids.device)
                input_ids = torch.where(cond < self.p_uncond, uncond_ids, input_ids)

            embeddings = self.encoder(input_ids)  # [B, L] -> [B, L, D]
            if mode == "eval":
                uncond_embeddings = self.encoder(uncond_ids)

        elif self.encoder_type == "GraphEncoder":
            node_types = [s.node_types for s in targets]
            edge_types = [s.edge_types for s in targets]
            edges = [s.edges for s in targets]
            batch = [(nt, et, e) for nt, et, e in zip(node_types, edge_types, edges)]

            objs, triples, obj_to_img, _ = graph_preprocesser(batch)
            objs = objs.to(input_ids.device)
            triples = triples.to(input_ids.device)
            obj_to_img = obj_to_img.to(input_ids.device)

            cat_emb = self.encoder(objs, triples, obj_to_img)
            _embeddings = [cat_emb[obj_to_img == i] for i in range(B)]
            embeddings, valid_mask, length = pad_embeddings(_embeddings)

            B2, Lmax, D = embeddings.shape
            assert B2 == B

            uncond_valid_mask = torch.zeros(
                (B, Lmax), device=embeddings.device, dtype=torch.bool
            )
            uncond_valid_mask[:, 0] = True

            null = self.graph_null_emb.to(embeddings)[None, None, :]  # [1,1,D]
            null = null.expand(B, 1, -1)  # [B,1,D]
            _z_pad = embeddings.new_zeros(B, Lmax - 1, D)  # zero tensor [B, Lmax-1, D]
            uncond_embeddings = torch.cat([null, _z_pad], dim=1)  # [B, Lmax, D]

            if mode == "train":
                assert self.training
                drop = torch.rand(B, device=input_ids.device) < self.p_uncond

                if drop.any():
                    embeddings = embeddings.clone()
                    valid_mask = valid_mask.clone()
                    embeddings[drop] = uncond_embeddings[drop]
                    valid_mask[drop] = uncond_valid_mask[drop]
            elif mode == "eval":
                assert not self.training
                # use uncond directly
                pass

        else:
            raise NotImplementedError(f"{self.encoder_type} is not implemented")

        # return (embeddings, mask), (uncond_embeddings, uncond_mask)
        return (embeddings, valid_mask), (uncond_embeddings, uncond_valid_mask)

    def _train(self, targets):
        imgs_list = [s.img for s in targets]
        labels = [s.tex for s in targets]
        uncond_labels = [["<MASK>"] for s in targets]

        imgs = torch.stack(imgs_list)  # [B, C, H, W]
        input_ids = self.tokenizer(labels, max(map(len, labels)))
        uncond_ids = self.tokenizer(uncond_labels, input_ids.shape[1])

        (embeddings, valid_mask), (_, _) = self._encode(
            input_ids, targets, uncond_ids=uncond_ids, mode="train"
        )

        with torch.no_grad():
            latent_dist = self.vae.encode(imgs)
            latents = latent_dist.latent_dist.sample()

            latents = latents * self.vae.config.scaling_factor

        # Sample noise to add to the images
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
            dtype=torch.int64,
        )

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        if self.encoder_type == "GraphEncoder":
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=embeddings,
                encoder_attention_mask=valid_mask,  # [B, L],  `True` the mask is kept, otherwise if `False` it is discarded
            ).sample
        else:
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=embeddings,
            ).sample

        return dict(loss=F.mse_loss(noise_pred, noise))

    def _valid(self, targets):
        num_inference_steps = 25
        height = 224
        width = 224

        batch_size = len(targets)

        labels = [s.tex for s in targets]
        uncond_labels = [["<MASK>"] for s in targets]
        input_ids = self.tokenizer(labels, max(map(len, labels)))  # [B, ] -> [B, L]
        uncond_ids = self.tokenizer(
            uncond_labels, input_ids.shape[1]
        )  # [B, ] -> [B, L]

        captions = [" ".join(labe) for labe in labels]

        with torch.no_grad():
            (embeddings, valid_mask), (uncond_embeddings, uncond_valid_mask) = (
                self._encode(input_ids, targets, uncond_ids=uncond_ids, mode="eval")
            )

        embeddings = torch.cat([uncond_embeddings, embeddings])

        if self.encoder_type == "GraphEncoder":
            valid_mask = torch.cat([uncond_valid_mask, valid_mask])
        elif self.encoder_type == "TextEncoder":
            # this encoder type do not use masks for Unet condition
            pass
        else:
            raise NotImplementedError(
                f"{self.encoder_type} not defined about mask usage."
            )

        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
        ).to(input_ids.device)

        latents = latents * self.noise_scheduler.init_noise_sigma
        self.noise_scheduler.set_timesteps(num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                if self.encoder_type == "GraphEncoder":
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=embeddings,
                        encoder_attention_mask=valid_mask,  # [B, L],  `True` the mask is kept, otherwise if `False` it is discarded
                    ).sample
                else:
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=embeddings
                    ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return [
            dict(image=image, caption=caption, input=tgt)
            for image, caption, tgt in zip(image.detach().cpu(), captions, targets)
        ]
