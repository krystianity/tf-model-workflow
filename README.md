# Tensorflow 2.0 Model Workflow

* Setting up the environment via conda
* Table to dataset in minutes via feature_columns
* Model compilation via tf.keras interface
* Hyperparameter tuning via kerastuner (not supported currently https://github.com/keras-team/keras-tuner)
* Model export
* TFX (Tensorflow Serving) scalable model deployment

## Setup

```bash
brew install python3
open https://www.anaconda.com/distribution/#macos
# download the installer for python 3.7 and install it
nano ~/.zshrc
export PATH=/anaconda3/bin:$PATH
# restart shell
conda init zsh
# restart shell
```

## Creating the environment

```bash
cd tf-model-workflow
conda create -n tfmw python=3.6 pip
conda activate tfmw
pip install tensorflow==2.0.0-beta1
pip install numpy pandas sklearn
# conda deactivate # in case of leaving
```

## Compiling, Training, Tuning and Exporting the model

```bash
python train.py
```