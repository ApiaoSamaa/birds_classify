# Bird Classification Assignment


### 1.0. Package

* install requirements
* replace folder timm/ to our timm/ folder (for ViT or Swin-T)
  #### pytorch model implementation [timm](https://github.com/rwightman/pytorch-image-models)

  #### recommand [anaconda](https://www.anaconda.com/products/distribution)

  #### recommand [weights and biases](https://wandb.ai/site)

  #### [deepspeed](https://www.deepspeed.ai/getting-started/) // future works

### 1.1. Dataset

According to the assignmnent:

* [Part of the CUB-200-2011](https://pan.seu.edu.cn:443/link/287D50CDC924503CD214A25811B3D1D8)

### 1.2. OS

- [X] Windows10
- [X] Ubuntu20.04
- [X] macOS (CPU only)

## 2. Train

- [X] Single GPU Training
- [ ] DataParallel (single machine multi-gpus)
- [ ] DistributedDataParallel

(more information: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### 2.1. data

train data and test data structure:

```
├── tain/
│   ├── class1/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   ├── class2/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   └── ....
└──
```

### 2.2. configuration

you can directly modify yaml file (in ./configs/)

### 2.3. run

```
python main.py --c ./configs/config.yaml
```

model will save in ./records/{project_name}/{exp_name}/backup/

If you only want to see the test class result, please run:

```
python run_results.py --c ./configs/config_assign.yaml
clean_result.bash # this will make your file sorted according to the first column
```

And then, check your records/ dir, you can find sorted_eval_results.txt under your experiment directory.