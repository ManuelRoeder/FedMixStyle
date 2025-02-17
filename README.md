
# [FedMixStyle] Efficient Cross-Domain Federated Learning by MixStyle Approximation

This repository contains the prototype code for FedMixStyle, an efficient cross-domain
FL framework employing MixStyle approximation for local client adaptation.

Our paper is presented at the ECML-PKKD 2023 workshop "Adapting to Change: Reliable Multimodal Learning Across Domains".

## Prerequisites
The project is optimized to run with Python 3.8.12+
Install Dassl.pytorch lib with:
```sh
cd Dassl.pytorch
python setup.py develop
```

## Dependency Installation

Fetch the dependencies for FedMixStyle.
```sh
pip install -r requirements.txt
```
Install CUDA drivers as needed.

## Data sets
Obtain data sets by following Dassl.pytorch data set README:
https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md
```

## Server-Client Deployment
Since the server-client architecture of FedMixStyle is build upon Flower, follow their installation instructions for on-device deployment (e.g. on Nvidia Jetson devices or RaspberryPis)
```
https://flower.dev/
https://flower.dev/blog/2020-12-16-running_federated_learning_applications_on_embedded_devices_with_flower/
```

## Need help? Did I forget to give credits? Please contact me -
manuel.roeder@thws.de - I am glad to help

## Ack
Flower - https://flower.dev/
LCCS - https://github.com/zwenyu/lccs
MixStyle - https://github.com/KaiyangZhou/mixstyle-release
Dassl.pytorch - https://github.com/KaiyangZhou/Dassl.pytorch

