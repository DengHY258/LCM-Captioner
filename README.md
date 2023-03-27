# LCM-Captioner: A Lightweight Text-based Image Captioning method with Collaborative Mechanism between Vision and Text
LCM-Captioner is an efficient model for Text-based Image Captioning(TextCap). This repository contains the code for LCM-Captioner model, released under the Pythia framework.
> [**LCM-Captioner: A Lightweight Text-based Image Captioning method with Collaborative Mechanism between Vision and Text**]() \
> Qi Wang\*, Hongyu Deng\*, Xue Wu, Zhenguo Yang, Yun Liu, Yazhou Wang, Gefei Hao \
> \* equal contribution \


<p align="center"> <img src='docs/overview.png' align="center"> </p>


## Installation
Clone this repository, and build it with the following command.
```
cd ~/pythia
python setup.py build develop
# install pycocoevalcap
# use the repo below instead of https://github.com/tylin/coco-caption
# note: you also need to have java on your machine
pip install git+https://github.com/ronghanghu/coco-caption.git@python23
```

## Getting Data
The dataset used by LCM-Captioner is TextCaps([PDF](https://arxiv.org/pdf/2003.12462.pdf)). You can download the vocabulary files, imdbs and features from the links below. 

### Vocabulary Files
[Vocabs](https://dl.fbaipublicfiles.com/pythia/m4c_captioner/data/m4c_captioner_vocabs.tar.gz) 
### TextCaps ImDB
[TextCaps ImDB](https://dl.fbaipublicfiles.com/pythia/m4c_captioner/data/imdb/m4c_textcaps.tar.gz)
### Object Faster R-CNN Features
[Object Feature](https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz)
### OCR Faster R-CNN Features
[OCR Feature](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_textvqa_ocr_en_frcn_features.tar.gz)


**Note that imdbs should be extracted under `data/imdb/`. All other files should be extracted under `data/`.**

### Detectron Weight
Please follow the instructions below to download the detectron weight for Faster RCNN
```
# Download detectron weights
cd data/
wget http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf detectron_weights.tar.gz
cd ..
```

## Training
There are two ways to train the LCM-Captioner model on the TextCaps training set:
1) to train on a multi-GPUs machine:
```
python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/LCM_captioner.yml \
--save_dir save/LCM_captioner \
training_parameters.distributed True
# change `--nproc_per_node 4` to the actual GPU number on your machine
```

2) to train on a single-GPU machine:
```
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/LCM_captioner.yml \
--save_dir save/LCM_captioner \
training_parameters.data_parallel True
```

## Evaluating
There are two steps to evaluate the trained model.
First, generate the prediction file with the trained snapshots:
```
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/LCM_captioner.yml \
--save_dir save/LCM_captioner \
--run_type val --evalai_inference 1 \
--resume_file save_model_path
```
Second, evaluate the prediction file:
```
python projects/M4C_Captioner/scripts/textcaps_eval.py --set val --pred_file prediction_file_path
```

## Test
You can generate the prediction file on TextCaps test split and submit it to the TextCaps EvalAI server.
```
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/LCM_captioner.yml \
--save_dir save/LCM_captioner \
--run_type inference --evalai_inference 1 \
--resume_file save_model_path
```


For more details, please follow the [file]('projects/M4C_Captioner/README.md').

## Pretrained Model
[Google Drive](https://drive.google.com/file/d/17k8KnlDkj90Zr4F-MBu1RyWq6OwVNMp8/view?usp=sharing)

## Citation

If our work is helpful to you, please cite:
```
@article{wang2023lcm,
  title={LCM-Captioner: A lightweight text-based image captioning method with collaborative mechanism between vision and text},
  author={Wang, Qi and Deng, Hongyu and Wu, Xue and Yang, Zhenguo and Liu, Yun and Wang, Yazhou and Hao, Gefei},
  journal={Neural Networks},
  year={2023},
  publisher={Elsevier}
}
```

