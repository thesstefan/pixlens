import logging
from pathlib import Path

import kornia.augmentation as K
import torch
from PIL import Image
from torch import nn, optim
from torch.cuda import get_device_properties
from torchvision import transforms
from torchvision.transforms import functional as TF

from pixlens.editing import interfaces
from pixlens.editing.impl.vqgan_clip.CLIP import clip
from pixlens.editing.impl.vqgan_clip.helper_generate import (
    ClampWithGrad,
    Prompt,
    ReplaceGrad,
    load_vqgan_model,
    resample,
    vector_quantize,
)
from pixlens.editing.utils import (
    generate_simplified_description_based_prompt,
)
from pixlens.evaluation.interfaces import Edit
from pixlens.evaluation.preprocessing_pipeline import PreprocessingPipeline
from pixlens.utils.utils import get_cache_dir

default_image_size = 512  # >8GB VRAM
if not torch.cuda.is_available():
    default_image_size = 256  # no GPU found
elif (
    get_device_properties(0).total_memory <= 2**33
):  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    default_image_size = 304  # <8GB VRAM


class MakeCutouts(nn.Module):
    def __init__(
        self,
        augments: list,
        cut_size: int,
        cutn: int,
        cut_pow: float = 1.0,
    ) -> None:
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow  # not used with pooling
        self.noise_fac = 0.1
        # Pick your own augments & their order
        augment_list = []
        for item in augments[0]:
            if item == "Ji":
                augment_list.append(
                    K.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.1,
                        p=0.7,
                    ),
                )
            elif item == "Pe":
                augment_list.append(
                    K.RandomPerspective(distortion_scale=0.7, p=0.7),
                )

            elif item == "Af":
                augment_list.append(
                    K.RandomAffine(
                        degrees=15,
                        translate=0.1,
                        shear=5,
                        p=0.7,
                        padding_mode="zeros",
                        keepdim=True,
                    ),
                )  # border, reflection, zeros

            elif item == "Er":
                augment_list.append(
                    K.RandomErasing(
                        scale=(0.1, 0.4),
                        ratio=(0.3, 1 / 0.3),
                        same_on_batch=True,
                        p=0.7,
                    ),
                )
            self.augs = nn.Sequential(*augment_list)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        sidey, sidex = input.shape[2:4]
        max_size = min(sidex, sidey)
        min_size = min(sidex, sidey, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size)
                + min_size,
            )
            offsetx = torch.randint(0, sidex - size + 1, ())
            offsety = torch.randint(0, sidey - size + 1, ())
            cutout = input[
                :,
                :,
                offsety : offsety + size,
                offsetx : offsetx + size,
            ]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(
                0,
                self.noise_fac,
            )
            batch = batch + facs * torch.randn_like(batch)
        return batch


class VqGANClip(interfaces.PromptableImageEditingModel):
    device: torch.device | None

    def __init__(
        self,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        self.prompts = None
        self.max_iterations = 500
        self.display_freq = 50
        self.size = [
            default_image_size,
            default_image_size,
        ]
        self.init_image = None
        self.init_noise = None
        self.init_weight = 0.0
        self.clip_model = "ViT-B/32"
        self.vqgan_config = Path(
            get_cache_dir()
            / "models--VqGANClip/checkpoints/vqgan_imagenet_f16_16384.yaml",
        )
        self.vqgan_checkpoint = Path(
            get_cache_dir()
            / "models--VqGANClip/checkpoints/vqgan_imagenet_f16_16384.ckpt",
        )
        if not Path.exists(self.vqgan_config):
            logging.info(
                "Download VQGAN-CLIP model",
            )
        self.noise_prompt_seeds = []
        self.noise_prompt_weights = []
        self.step_size = 0.1
        self.cut_method = "latest"
        self.cutn = 32
        self.cut_pow = 1.0
        self.seed = seed
        self.optimiser = "Adam"
        self.zoom_start = 0
        self.zoom_frequency = 10
        self.zoom_scale = 0.99
        self.zoom_shift_x = 0
        self.zoom_shift_y = 0
        self.prompt_frequency = 0
        self.video_length = 10
        self.output_video_fps = 0
        self.input_video_fps = 15
        self.cudnn_determinism = False
        self.augments = [["Af", "Pe", "Ji", "Er"]]
        self.video_style_dir = None
        self.device = device
        self.replace_grad = ReplaceGrad.apply
        self.clamp_with_grad = ClampWithGrad.apply
        if self.seed is None:
            self.seed = torch.seed()
        torch.manual_seed(self.seed)
        self.model = load_vqgan_model(
            self.vqgan_config,
            self.vqgan_checkpoint,
        ).to(self.device)
        jit = "1.7.1" in torch.__version__
        self.perceptor = (
            clip.load(self.clip_model, jit=jit)[0]
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        self.cut_size = self.perceptor.visual.input_resolution
        self.f = 2 ** (self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(
            self.augments,
            self.cut_size,
            self.cutn,
            cut_pow=self.cut_pow,
        )
        toksx, toksy = self.size[0] // self.f, self.size[1] // self.f
        self.sidex, self.sidey = toksx * self.f, toksy * self.f
        self.e_dim = self.model.quantize.e_dim
        self.n_toks = self.model.quantize.n_e
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[
            None,
            :,
            None,
            None,
        ]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[
            None,
            :,
            None,
            None,
        ]
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "seed": self.seed,
        }

    def split_prompt(self, prompt: str) -> tuple[str, float, float]:
        vals = prompt.rsplit(":", 2)
        vals = vals + ["", "1", "-inf"][len(vals) :]
        return vals[0], float(vals[1]), float(vals[2])

    def get_opt(
        self,
        z: torch.Tensor,
        opt_name: str,
        opt_lr: float,
    ) -> optim.Optimizer:
        if opt_name == "Adam":
            opt = optim.Adam([z], lr=opt_lr)  # LR=0.1 (Default)
        else:
            msg = f"Unknown optimizer: {opt_name}"
            raise ValueError(msg)
        return opt

    def prepare_image(
        self,
        image_path: str,
    ) -> torch.Tensor:
        img = Image.open(image_path)
        pil_image = img.convert("RGB")
        pil_image = pil_image.resize((self.sidex, self.sidey), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = self.model.encode(
            pil_tensor.to(self.device).unsqueeze(0) * 2 - 1,
        )
        return z

    def synth(self, z: torch.Tensor) -> torch.Tensor:
        z_q = vector_quantize(
            z.movedim(1, 3),
            self.model.quantize.embedding.weight,
        ).movedim(3, 1)
        return self.clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    def ascend_txt(self) -> list[torch.Tensor]:
        out = self.synth(self.z)
        iii = self.perceptor.encode_image(
            self.normalize(self.make_cutouts(out)),
        ).float()
        return [prompt(iii) for prompt in self.pms]

    def train(self) -> None:
        self.opt.zero_grad(set_to_none=True)
        loss_all = self.ascend_txt()
        loss = sum(loss_all)
        loss.backward()
        self.opt.step()
        with torch.inference_mode():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        del edit_info
        self.pms = []
        self.z = self.prepare_image(image_path)
        self.z.requires_grad_(True)  # noqa: FBT003
        txt, weight, stop = self.split_prompt(prompt)
        embed = self.perceptor.encode_text(
            clip.tokenize(txt).to(self.device),
        ).float()
        self.pms.append(Prompt(embed, weight, stop).to(self.device))
        self.opt = self.get_opt(self.z, self.optimiser, self.step_size)
        for _ in range(self.max_iterations):
            self.train()
        return TF.to_pil_image(self.synth(self.z).squeeze(0).cpu())

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.DESCRIPTION

    def generate_prompt(self, edit: Edit) -> str:
        return generate_simplified_description_based_prompt(edit)

    def get_latent(self, prompt: str, image_path: str) -> torch.Tensor:
        self.pms = []
        self.z = self.prepare_image(image_path)
        self.z.requires_grad_(True)  # noqa: FBT003
        txt, weight, stop = self.split_prompt(prompt)
        embed = self.perceptor.encode_text(
            clip.tokenize(txt).to(self.device),
        ).float()
        self.pms.append(Prompt(embed, weight, stop).to(self.device))
        self.opt = self.get_opt(self.z, self.optimiser, self.step_size)
        for _ in range(self.max_iterations):
            self.train()
        return self.z


if __name__ == "__main__":
    model = VqGANClip(device=torch.device("cuda"), seed=42)
    preprocessing_pipe = PreprocessingPipeline(
        "./pixlens/editval/object.json",
        "./editval_instances/",
    )
    preprocessing_pipe.execute_pipeline([model])
