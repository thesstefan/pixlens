import logging
import pickle
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from pixlens.editing import interfaces
from pixlens.editing.impl.open_edit.options.opt import Config
from pixlens.editing.impl.open_edit.trainers.OpenEdit_optimizer import (
    OpenEditOptimizer,
)
from pixlens.editing.impl.open_edit.util.visualizer import Visualizer
from pixlens.editing.impl.open_edit.util.vocab import Vocabulary  # noqa: F401
from pixlens.editing.utils import (
    generate_original_description,
    generate_simplified_description_based_prompt,
)
from pixlens.evaluation.interfaces import Edit
from pixlens.evaluation.preprocessing_pipeline import PreprocessingPipeline


class OpenEdit(interfaces.PromptableImageEditingModel):
    device: torch.device | None

    def __init__(
        self,
        alpha: float = 5,
        optimize_iter: int = 10,
        device: torch.device | None = None,
    ) -> None:
        self.device = device

        opt = Config()
        if not Path.exists(opt.checkpoints_dir):
            logging.info(
                "Model not found in %s! Please download it first.",
                opt.checkpoints_dir,
            )
        opt.gpu = 0
        self.global_edit = True
        self.alpha = alpha
        self.optimize_iter = optimize_iter
        opt.world_size = 1
        opt.rank = 0
        opt.mpdist = False
        opt.num_gpu = 1
        opt.batchSize = 1
        opt.manipulation = True
        opt.perturbation = True
        self.opt = opt
        self.open_edit_optimizer = OpenEditOptimizer(opt)
        self.open_edit_optimizer.open_edit_model.netG.eval()
        self.visualizer = Visualizer(opt, rank=0)
        self.str_path_to_impl = "pixlens/editing/impl/open_edit/"

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "alpha": self.alpha,
            "optimize_iter": self.optimize_iter,
        }

    def image_loader(self, image_path: str) -> torch.Tensor:
        transforms_list = []
        transforms_list.append(
            transforms.Resize((self.opt.img_size, self.opt.img_size)),
        )
        transforms_list += [transforms.ToTensor()]
        transforms_list += [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform = transforms.Compose(transforms_list)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        if self.device == torch.device("cuda"):
            return image_tensor.unsqueeze(0).cuda()  # type: torch.Tensor
        return image_tensor.unsqueeze(0)

    def text_loader(
        self,
        ori_cap_str: str,
        new_cap_str: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ori_cap = ori_cap_str.split()
        new_cap = new_cap_str.split()
        vocab = pickle.load(  # noqa: S301
            Path.open(
                self.str_path_to_impl
                + "vocab/"
                + self.opt.dataset_mode
                + "_vocab.pkl",
                "rb",
            ),
        )
        ori_txt = (
            [vocab("<start>")]
            + [vocab(word) for word in ori_cap]
            + [vocab("<end>")]
        )

        new_txt = (
            [vocab("<start>")]
            + [vocab(word) for word in new_cap]
            + [vocab("<end>")]
        )
        if self.device == torch.device("cuda"):
            ori_txt_tensor = torch.LongTensor(ori_txt).unsqueeze(0).cuda()
            new_txt_tensor = torch.LongTensor(new_txt).unsqueeze(0).cuda()
            return ori_txt_tensor, new_txt_tensor
        return torch.LongTensor(ori_txt).unsqueeze(0), torch.LongTensor(
            new_txt,
        ).unsqueeze(0)

    def edit_image_with_optimized_perturbations(
        self,
        data: dict,
        ori_tensor: torch.Tensor,
        new_tensor: torch.Tensor,
    ) -> Image.Image:
        visuals = {}
        for _ in range(self.optimize_iter):
            self.open_edit_optimizer.run_opt_one_step(
                data,
                ori_tensor,
                new_tensor,
                self.alpha,
                global_edit=self.global_edit,
            )
        visuals[
            "optimized_manipulated"
        ] = self.open_edit_optimizer.open_edit_model(
            data,
            mode="manipulate",
            ori_cap=ori_tensor,
            new_cap=new_tensor,
            alpha=self.alpha,
        )[0]
        visuals = self.visualizer.convert_visuals_to_numpy(visuals, gray=True)
        return Image.fromarray(visuals["optimized_manipulated"])

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        input_image = self.image_loader(image_path)
        ori_tensor, new_tensor = self.text_loader(
            generate_original_description(edit_info),
            prompt,
        )
        data = {"image": input_image, "caption": new_tensor, "length": [4]}
        return self.edit_image_with_optimized_perturbations(
            data,
            ori_tensor,
            new_tensor,
        )

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.DESCRIPTION

    def generate_prompt(self, edit: Edit) -> str:
        return generate_simplified_description_based_prompt(edit)

    def get_latent(self, prompt: str, image_path: str) -> torch.Tensor:
        raise NotImplementedError
