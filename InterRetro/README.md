
# Interactive Retrosynthesis Planning (InterRetro)

This is the official code repository for NeurIPS 2025 paper *Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs*.

## Setup

The complete codebase is organised into three subfolders:

```
InterRetro
|- Graph2Edits
|- InterRetro
|- dataset
```

The `Graph2Edits` and `InterRetro` folders have already been built and initialised; no further action is required.
For the `dataset` folder, please download it from [this link](https://drive.google.com/drive/folders/198WuPlSyMeMvvd4i2SM833jPAcGllzDu).

## Usage

The required packages are recorded in `requirements.txt`. Please install them using the following command:
```
cd InterRetro
pip install -r requirements.txt
```

To train the InterRetro model with wandb logs, run the following command:
```
python InterRetro/main.py
```

To train the model without wandb logs, run the following command:
```
python InterRetro/main.py --enable_wandb 0
```

You can find available configuration options in the `config_parser.py` file.

## Reference

If you find our research helpful, please cite our paper at NeurIPS 2025:
```
@inproceedings{
  wang2025interretro,
  title={Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured {MDP}s},
  author={Mianchu Wang and Giovanni Montana},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=m7uj1vIZ62}
}
```

