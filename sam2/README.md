## Environment
```python=3.10.15, pytorch=2.5.1, torchvision=0.20.1```

```sh
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
pip install -e ".[notebooks]"
cd ../checkpoints && download_ckpts.sh
cd ..
```

## Generate Masks
```sh
python generate_masks.py
```