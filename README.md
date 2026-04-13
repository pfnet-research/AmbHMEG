# AmbHMEG: Ambiguous Handwritten Mathematical Expression Generator

This repository provides code to generate images of ambiguous handwritten mathematical expressions.
It includes scripts for model training and inference.

## Usage

### Install

Set up the Python environment and install all required dependencies:

```sh
uv sync --all-packages --locked
```

### Dataset

Please download the preprocessed data from [Hugging Face](https://huggingface.co/pfnet/amb-hmeg).
After downloading the dataset, extract it with the following command:

```sh
tar --zstd -xvf mathwriting_with_graph.tar.zst -C ~
```

### Training

To train on the MathWriting dataset, run:

```sh
python scripts/train.py
```

### Inference

Place the pretrained model `gryphgen_graph_sd_mathwriting_epoch_50.pth` in the AmbHMEG directory.

#### Normal Generation

```sh
python scripts/generate.py
```

#### Ambiguous Generation

```sh
python scripts/generate_gh.py --edit layout
python scripts/generate_gh.py --edit symbol
```

## License

### AmbHMEG

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for more details.

### MathWriting

This project makes use of the [MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting) dataset developed by Google Research.
We acknowledge and thank the original authors for providing this valuable resource.
The dataset is licensed under CC BY-NC-SA 4.0.
Users are responsible for complying with the terms of this license when using the dataset or any derived outputs.
