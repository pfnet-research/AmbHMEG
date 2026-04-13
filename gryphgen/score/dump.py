import math
import pickle
from pathlib import Path

import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MessageHub, MMLogger
from mmengine.registry import METRICS
from PIL import Image, ImageDraw, ImageFont


def _to_image(img_t: torch.Tensor) -> Image:
    """
    img_t: (C,H,W) float tensor.
    """
    if img_t.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img_t.shape)}")

    x = img_t.detach().cpu()

    x = x.clamp(0, 1)
    x = x >= 0.5
    x = x.to(torch.uint8)

    x = x.mul(255)
    x = x.permute(1, 2, 0)  # (H, W, C)

    return Image.fromarray(x.numpy())


def _to_uint8_pil(img_t: torch.Tensor) -> Image.Image:
    """
    img_t: (C,H,W) float tensor.
    """
    if img_t.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img_t.shape)}")

    x = img_t.detach().cpu()

    x = x.clamp(0, 1)

    x = (x * 255).to(torch.uint8)
    x = x.permute(1, 2, 0).contiguous()

    if x.shape[2] == 1:
        x = x.expand(x.shape[0], x.shape[1], 3)

    return Image.fromarray(x.numpy(), mode="RGB")


def _wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
):

    words = text.split()
    if not words:
        return [""]

    lines = []
    cur = words[0]
    for w in words[1:]:
        test = cur + " " + w
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def make_grid_with_titles(
    images: list[torch.Tensor],
    titles: list[str],
    nrow: int,
    padding: int = 8,
    title_h: int = 28,
    border_w: int = 2,
    bg=(255, 255, 255),
    border=(0, 0, 0),
    text_color=(0, 0, 0),
) -> Image.Image:

    assert len(images) == len(titles)
    N = len(images)
    if N == 0:
        raise ValueError("No images")

    pil_imgs = [_to_uint8_pil(t) for t in images]
    W, H = pil_imgs[0].size
    tile_w = W
    tile_h = title_h + H

    ncol = nrow
    nrows = (N + ncol - 1) // ncol

    canvas_w = padding + ncol * tile_w + (ncol - 1) * padding + padding
    canvas_h = padding + nrows * tile_h + (nrows - 1) * padding + padding

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i in range(N):
        r = i // ncol
        c = i % ncol
        x0 = padding + c * (tile_w + padding)
        y0 = padding + r * (tile_h + padding)

        draw.rectangle(
            [x0, y0, x0 + tile_w, y0 + tile_h],
            outline=border,
            width=border_w,
        )

        title = titles[i] if titles[i] is not None else ""

        max_text_w = tile_w - 8
        lines = _wrap_text(draw, title, font, max_text_w)[:2]

        ty = y0 + 4
        for line in lines:
            draw.text((x0 + 4, ty), line, fill=text_color, font=font)
            ty += 16

        canvas.paste(pil_imgs[i], (x0, y0 + title_h))

        draw.line(
            [(x0, y0 + title_h), (x0 + tile_w, y0 + title_h)],
            fill=border,
            width=border_w,
        )

    return canvas


@METRICS.register_module()
class DumpImage(BaseMetric):
    def __init__(
        self,
        output_dir: str = "dump",
        nrow=None,
        padding: int = 8,
        title_h: int = 28 * 2,
        border_w: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.nrow = nrow
        self.padding = padding
        self.title_h = title_h
        self.border_w = border_w
        self._counter = 0

    def _resolve_dump_dir(self) -> Path:
        logger = MMLogger.get_current_instance()
        log_file = getattr(logger, "log_file", None)

        if log_file:
            work_dir = Path(log_file).resolve().parent
        else:
            work_dir = Path.cwd().resolve()

        dump_dir = self.output_dir
        if not dump_dir.is_absolute():
            dump_dir = work_dir / dump_dir

        dump_dir.mkdir(parents=True, exist_ok=True)
        return dump_dir

    def _get_epoch(self):
        hub = MessageHub.get_current_instance()
        epoch = hub.get_info("epoch")
        return None if epoch is None else int(epoch)

    @torch.no_grad()
    def process(self, data_batch, data_samples, epoch=None):
        dump_dir = self._resolve_dump_dir()

        epoch = self._get_epoch() if epoch is None else epoch
        suffix = f"_epoch{(epoch + 1):04d}" if epoch is not None else ""

        images = [s["image"] for s in data_samples]
        captions = [s.get("caption", "") for s in data_samples]

        nrow = self.nrow
        if nrow is None:
            nrow = int(math.sqrt(len(images))) or 1

        idx = self._counter
        self._counter += 1

        img_path = dump_dir / f"eval{suffix}_batch{idx:06d}.png"
        cap_path = dump_dir / f"eval{suffix}_batch{idx:06d}.txt"

        grid = make_grid_with_titles(
            images=images,
            titles=captions,
            nrow=nrow,
            padding=self.padding,
            title_h=self.title_h,
            border_w=self.border_w,
        )
        grid.save(img_path)

        with open(cap_path, "w", encoding="utf-8") as f:
            for i, cap in enumerate(captions):
                row = i // nrow
                col = i % nrow
                f.write(f"{i}\trow={row}\tcol={col}\t{cap}\n")

    def compute_metrics(self, results):
        return {}


@METRICS.register_module()
class DumpData(BaseMetric):
    def __init__(
        self,
        output_dir: str = "dump",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)

    def _resolve_dump_dir(self) -> Path:
        logger = MMLogger.get_current_instance()
        log_file = getattr(logger, "log_file", None)

        if log_file:
            work_dir = Path(log_file).resolve().parent
        else:
            work_dir = Path.cwd().resolve()

        dump_dir = self.output_dir.expanduser()
        if not dump_dir.is_absolute():
            dump_dir = work_dir / dump_dir

        dump_dir.mkdir(parents=True, exist_ok=True)
        return dump_dir

    def _get_epoch(self):
        hub = MessageHub.get_current_instance()
        epoch = hub.get_info("epoch")
        return None if epoch is None else int(epoch)

    @torch.no_grad()
    def process(self, data_batch, data_samples):
        dump_dir = self._resolve_dump_dir()

        for s, b in zip(data_samples, data_batch["targets"]):
            image = _to_image(s["image"])

            caption = s["caption"]
            name = b.name
            tex = b.tex

            data = dict(name=name, tex=tex, caption=caption, generated=True)

            pkl_path = dump_dir / f"{name}.pkl"
            pkl_path.write_bytes(pickle.dumps(data))

            img_path = dump_dir / f"{name}.png"
            image.save(img_path)

    def compute_metrics(self, results):
        return {}
