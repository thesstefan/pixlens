# PixLens

Evaluate & understand image editing models!

# Installation & Deveolpment Setup

## Venv

`python>=3.11` is required when using [`venv`](https://docs.python.org/3/library/venv.html).

```shell
# Create venv env
python -m venv $ENVNAME

# Activate env
source $ENVNAME/bin/activate

# Install in editable mode (remove [dev] if dev packages are not needed)
pip install -e ".[dev]"
```
## Conda

```shell
# Create conda env
conda create -n $ENVNAME "python>=3.11" --file requirements.txt -c pytorch -c nvidia -c conda-forge

# Install dev packages if needed
conda install --name $ENVNAME --freeze-installed --file requirements-dev.txt

# Activate environment
conda activate $ENVNAME

# Install in editable mode
pip install --no-build-isolation --no-deps -e .
```

## Installation

Once your environment is set up, you will need modify the `taming.modules.vqvae.quantize` module. This integration is crucial for the proper functioning of diffedit. The detailed instructions for this integration are also available in this [issue](https://github.com/CompVis/stable-diffusion/issues/72#issuecomment-1224675757).
