import pickle
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from pixlens.editing.impl.open_edit.options.train_options import TrainOptions
from pixlens.editing.impl.open_edit.trainers.OpenEdit_optimizer import (
    OpenEditOptimizer,
)
from pixlens.editing.impl.open_edit.util.visualizer import Visualizer


class OpenEditEditingModel:
    def __init__(self) -> None:
        opt = TrainOptions().parse()
        opt.gpu = 0
        self.global_edit = False
        self.alpha = 5
        self.optimize_iter = 10
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

    def image_loader(self, image_path: str) -> torch.Tensor:
        transforms_list = []
        transforms_list.append(
            transforms.Resize((self.opt.img_size, self.opt.img_size))
        )
        transforms_list += [transforms.ToTensor()]
        transforms_list += [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(transforms_list)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0).cuda()

    def text_loader(
        self, ori_cap_str: str, new_cap_str: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ori_cap = ori_cap_str.split()
        new_cap = new_cap_str.split()
        vocab = pickle.load(  # noqa: S301
            Path.open("vocab/" + self.opt.dataset_mode + "_vocab.pkl", "rb"),
        )
        ori_txt = (
            [vocab("<start>")]
            + [vocab(word) for word in ori_cap]
            + [vocab("<end>")]
        )
        ori_txt_tensor = torch.LongTensor(ori_txt).unsqueeze(0).cuda()

        new_txt = (
            [vocab("<start>")]
            + [vocab(word) for word in new_cap]
            + [vocab("<end>")]
        )
        new_txt_tensor = torch.LongTensor(new_txt).unsqueeze(0).cuda()
        return ori_txt_tensor, new_txt_tensor

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
        # edit_info: Edit | None = None,)
        ori_cap: str,  # it would be the category
    ) -> Image.Image:
        input_image = self.image_loader(image_path)
        ori_tensor, new_tensor = self.text_loader(ori_cap, prompt)
        data = {"image": input_image, "caption": new_tensor, "length": [4]}
        return self.edit_image_with_optimized_perturbations(
            data,
            ori_tensor,
            new_tensor,
        )
