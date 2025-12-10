# Neural Vocoder

## About

This repository contains a an implementation of HiFi-GAN from the original [paper](https://arxiv.org/pdf/2010.05646).

## Installation

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

To train a model, run the following command:

```bash
python train.py model=hifi_gan_v2
```

To run inference (synthesize new audio):

```bash
python synthesize.py inferencer.from_pretrained={pretrained_model_path}
```

Where pretrained model can be downloaded from [huggingface](https://huggingface.co/ArthurGaleev/HiFi-GAN-v2/blob/main/HiFi-GAN-v2.3-epoch35.pth).

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
