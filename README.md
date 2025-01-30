# Diffusion Model from Scratch

This project implements a diffusion model from scratch using TensorFlow. The model is trained on the MNIST dataset and uses a U-Net architecture for the CNN network.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Sampling](#sampling)
- [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/rahulverma5/diffusion-model.git
    cd diffusion-model
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To train the model, run:
```sh
python main.py
```
## Project Structure

diffusion-model/
├── configs/
│   └── config.py          # Configuration file
├── data/
│   └── load_data.py       # Data loading and preprocessing
├── diffusion/
│   └── forward_diffusion.py # Forward diffusion process
├── models/
│   └── unet.py            # U-Net model definition
├── training/
│   ├── train.py           # Training loop
│   └── sample.py          # Sampling from the model
├── main.py                # Main script to run the training
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation



## Acknowledgements
This work is a part of the coursework AI3603 Computer Vision in IIT Hyderabad.
