# Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network.

[Paper](https://arxiv.org/abs/2510.04871)

### Requirements

- pytorch 2.8.0 or similar

```bash
git clone https://github.com/c-nnr/TinyRecursiveModels
cd TinyRecursiveModels
uv pip uninstall tensorflow # important for kaggle/colab machines
uv pip install "packaging" "ninja" "wheel" "setuptools" "setuptools-scm"
uv pip install -q --no-cache-dir --no-build-isolation "adam-atan2"
uv pip install -r requirements.txt
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

```bash
# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training evaluation training2 evaluation2 concept \
  --test-set-name evaluation2
```

*Runtime:* ~30mins and requires 35GB RAM to run

## Experiments

### ARC-AGI (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc12concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc12concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
global_batch_size=768 \
+run_name=${run_name} ema=True
```

*Runtime:* ~6 days


## Reference

If you find our work useful, please consider citing:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).
