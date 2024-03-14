# PixLens

Evaluate & understand image editing models!

## Installation

To set up the required Python 3.11 conda environment, run the following commands:

```shell
# Clone repository & cd into it
git clone https://github.com/thesstefan/pixlens && cd pixlens

# Download EditVal selected images and unzip them in "editval_instances"
# You can do this manually or by running gdown:
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1q_V2oxtGUCPE2vkQi88NvnGurg2Swf9N" -O editval_instances.zip
unzip editval_instances.zip && rm editval_instances.zip

# Create conda env
conda create -n $ENVNAME "python>=3.11" --file requirements.txt -c pytorch -c nvidia -c conda-forge

# Install dev packages if needed
conda install --name $ENVNAME --freeze-installed --file requirements-dev.txt

# Activate environment
conda activate $ENVNAME

# Install xformers==0.0.23.post1
# No available conda package [issue](https://github.com/facebookresearch/xformers/issues/749)
pip install xformers==0.0.23.post1

# Install Image Reward
pip install image-reward

# With the usage of vqgan+clip, one has to follow the instructions in the repo we are based on (https://github.com/nerdyrodent/VQGAN-CLIP), but essentially one can download:
pip install kornia==0.7.2 taming-transformers git+https://github.com/openai/CLIP.git

# When doing so, please change 'pytorch_lightning.utilities.distributed' for 'pytorch_lightning.utilities.rank_zero' in taming/main.py as indicated in the [issue](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11458#issuecomment-1609900319)

# Install pixlens in editable mode
pip install --no-build-isolation --no-deps -e .
```

The `editval_instances.zip` can also be downloaded from [here](https://drive.google.com/uc?export=download&id=1q_V2oxtGUCPE2vkQi88NvnGurg2Swf9N).

> **NOTICE 1**: The NullTextInversion model is available only when `diffusers=10.0.0` due to 
> issues when using newer versions ([#1](https://github.com/google/prompt-to-prompt/issues/57), 
> [#2](https://github.com/google/prompt-to-prompt/issues/72), [#3](https://github.com/google/prompt-to-prompt/issues/37)).
> Therefore, use the [`requirements-null-text-inv.txt`](https://github.com/thesstefan/pixlens/blob/main/requirements-null-text-inv.txt) 
> dependencies instead when dealing with NullTextInversion.
>
> **NOTICE 2**: On some bare-bones systems (like containers), it may be required to provide the `libgl1.so` dependency for OpenCV. The
> following error is raised be `pixlens` if the dependency is not available:
>```python
> ImportError: libGL.so.1: cannot open shared object file: No such file or directory
>```
> On Debian-based systems this can be installed by running
>```
> apt-get install libgl1
>```

## Evaluation

The models that are already implemented are InstructPix2Pix, ControlNet, LCM, DiffEdit & NullTextInversion. 
Due to some issues, DiffEdit and NullTextInversion are left out the full benchmark by default. 

The models that are available for detection & segmentation are GroundedSAM (SAM on top of GroundingDino) and 
Owl-ViTSAM (SAM on top of Owl-ViT).

All models can be loaded by specifying their corresponding YAML configurations 
from [`model_cfgs`](https://github.com/thesstefan/pixlens/tree/main/model_cfgs).

Edit operations that can currently be evaluated include:
- Object Addition
- Object Replacement
- Object Removal
- Part Alteration
- Moving Objects
- Positional Object Addition
- Size Change
- Color Change

Along these, subject & background preservation is evaluated for each edit.

Generally, you can expect to find some artifacts (edited images, segmentation results, explanatory visualization, scores) 
in PixLens's cache directory (`~/.cache/pixlens` on UNIX, `C:\Users\{USER}\AppData\Local\pixlens\pixlens/Cache` on Windows).

If the model VQGAN+CLIP is used, please download their checkpoints folder and place it in the PixLens's cache directory under the folder models--VqGANClip (in the end the folder Cache/models--VqGANClip/checkpoints should be there)-

If the model OpenEdit is used, also download their checkpoints and vocab folders, as indicated in the [repo](https://github.com/xh-liu/Open-Edit) and again place them under the folder models--openedit (in the end the folder Cache/models--openedit/vocab and .../checkpoints should be there).

###  Scripts

To run the whole evaluation pipeline (for InstructPix2Pix, ControlNet, LCM, OpenEdit and VQGAN+clip), run 
```shell
pixlens-eval --detection-model-yaml ${DETECTION_MODEL_YAML} --run-evaluation-pipeline
```

To run a more specific evaluation (for one specific model & operation type), run
```shell
pixlens-eval --detection-model-yaml ${DETECTION_MODEL_YAML} 
             --editing-model-yaml ${EDITING_MODEL_YAML}
             --edit-type ${EDIT_TYPE}
             --do-all
```

The results will be available in the mentioned cache directory under `evaluation_results.json` (aggregated) and
`evaluation_results.csv` (individual edits).

>Currently, here are the possible parameterizations:
>- `EDITING_MODEL_YAML` values: 
>[`model_cfgs/lcm.yaml`](https://github.com/thesstefan/pixlens/blob/main/model_cfgs/lcm.yaml),
>[`model_cfgs/instruct_pix2pix.yaml`](https://github.com/thesstefan/pixlens/blob/main/model_cfgs/instruct_pix2pix.yaml),
>[`model_cfgs/controlnet.yaml`](https://github.com/thesstefan/pixlens/blob/main/model_cfgs/controlnet.yaml),
>[`model_cfgs/null_text_inversion.yaml`](https://github.com/thesstefan/pixlens/blob/main/model_cfgs/null_text_inversion.yaml),
>[`model_cfgs/diffedit.yaml`](https://github.com/thesstefan/pixlens/blob/main/model_cfgs/diffedit.yaml),
>- `DETECTION_MODEL_YAML` values: [`model_cfgs/grounded_sam.yaml`](https://github.com/thesstefan/pixlens/blob/main/model_cfgs/grounded_sam.yaml)
>[`model_cfgs/owlvit_sam.yaml`](https://github.com/thesstefan/pixlens/blob/main/model_cfgs/owlvit_sam.yaml)
>- `EDIT_TYPE` values: `object_addition`, `object_replacement`, `object_removal`, `alter_parts`, 
>`position_replacement`, `positional_addition`, `size`, `color`

Similarly, there are other CLI scripts provided for debugging intermediary steps, like [`pixlens_editing`](https://github.com/thesstefan/pixlens/blob/main/pixlens/cli/pixlens_editing_cli.py),
[`pixlens_detection`](https://github.com/thesstefan/pixlens/blob/main/pixlens/cli/pixlens_detection_cli.py), or [`pixlens_caption`](https://github.com/thesstefan/pixlens/blob/main/pixlens/cli/pixlens_caption_cli.py).

## Disentanglement Pipeline

To execute the disentanglement pipeline, use the following command:

```shell
pixlens-disentanglement --model-params-yaml ${MODEL_PARAMS_YAML}
```
Upon completion of the pipeline, a folder titled `disentanglement` will be created within the model's cache directory. This folder contains critical outputs of the evaluation:

- `results.json`: A file that details the most significant findings of the evaluation.
- **Confusion Matrix Plot**: A visual representation to help understand the performance of the model.

In case you prefer not to rerun the entire process, you have the option to delete the `.pkl` files. These files store essential data required for the evaluation.


### Benchmarking Custom Models

You can also benchmark your own model by defining an adapter class similar to the ones in [`editing`](https://github.com/thesstefan/pixlens/tree/main/pixlens/editing), implementing
the [`PrompatbleImageEditingModel`](https://github.com/thesstefan/pixlens/blob/main/pixlens/editing/interfaces.py#L16) protocol.

Afterwards, define a YAML configuration file similar to the ones in [`model_cfgs`](https://github.com/thesstefan/pixlens/tree/main/model_cfgs) and use this file as the
parameter for the `--editing_model_yaml` flag of `pixlens-eval` and `pixlens-disentanglement`.

# Acknowledgements

The [NullTextInversion](https://arxiv.org/abs/2211.09794) implementation is from [google/prompt-to-prompt](https://github.com/google/prompt-to-prompt). Otherwise,
the other models ([GroundingDino](https://arxiv.org/abs/2303.05499), [OwlViT](https://arxiv.org/abs/2205.06230),  ([SAM](https://arxiv.org/abs/2304.02643),
[InstructPix2Pix](https://arxiv.org/abs/2211.09800), [LCM](https://arxiv.org/abs/2310.04378), [ControlNet](https://arxiv.org/abs/2302.05543),
[DiffEdit](https://arxiv.org/abs/2210.11427)) are provided through their own packages (`sam`, `groundingdino`) or HuggingFace.

PixLens aims to build onto [EditVal](https://github.com/deep-ml-research/editval_code), so some inspiration is taken from it.
pixlens-disentanglement --model-params-yaml ${EDITING_MODEL_YAML}
