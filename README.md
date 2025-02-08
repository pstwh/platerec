# platerec

<p align="center">
  <img src="https://raw.githubusercontent.com/pstwh/platerec/main/examples/example.jpg" width="768" />
</p>

`platerec` is a lightweight package for reading license plates using an ONNX model. It is designed to be part of a pipeline for detecting, cropping, and reading license plates. The underlying model is a mobilenetv2 as encoder and a light gpt for decoder. The training data comprises primarily Brazilian license plates, sourced from internet images, also synthetic data generated in the same font with transforms. The model repository can be found [here](https://github.com/pstwh/platerec-model).

Now it supports reading from different countries.
Currently:
- \[br]: Brazil
- \[us]: United States
- \[ue]: European Union
- \[ru]: Russia

The license plate recognition model is continuously being improved. However, accuracy can be significantly enhanced by fine-tuning the model with a dataset specific to your needs. We encourage you to explore the model training repository to learn how to build a customized model with increased accuracy for your format.

The first token after '<' will be the country plate type.

Example:
'<[br]ZZZ1Z11>'

[Video example](https://www.youtube.com/embed/8zyAIp5qGBA)

## Installation

To install the required dependencies, use the following command:

For cpu

```bash
pip install "platerec[cpu]"
```

For cuda
```bash
pip install "platerec[gpu]"
```

## Usage

### Command Line Interface

You can use the command line interface to detect license plates in an image:

```bash
platerec image_path [--encoder_path ENCODER_PATH] [--decoder_path DECODER_PATH] [--return_types RETURN_TYPE] [--providers PROVIDERS] [--no_platedet]
```

#### Arguments

- `image_path`: Path to the input image. Could be more than one image.
- `--encoder_path`: Path to the ONNX encoder model (default: `artifacts/encoder.onnx`).
- `--decoder_path`: Path to the ONNX decoder model (default: `artifacts/decoder.onnx`).
- `--tokenizer_path`: Path to the tokenizer json file (default: `artifacts/tokenizer.json`).
- `--return_type`: Output formats (choices: `word`, `char`). Word return the plate text and confidence detected, char return the plate chars detected with confidences for each char.
- `--providers`: ONNX Runtime providers (default: `CPUExecutionProvider`).
- `--no_platedet`: Not use platedet to detect plates first.

### Example

To just read an already cropped image:

```bash
python3 platerec/cli.py examples/1.jpg --return_type word
```

To detect license plates and read them:

```bash
python3 platerec/cli.py examples/1.jpg --return_type word
```

### Using in Code

To just read an already cropped image:

```python
from PIL import Image
from platerec import Platerec

platerec = Platerec()
image = Image.open('examples/1.jpg')
pred = platerec.read(image)
```
pred will be something like:
```
{'word': '[br]ZZZ1Z11', 'confidence': 0.98828125}
```

To detect license plates and read them:

```python
from PIL import Image
from platerec import Platerec

platerec = Platerec()
image = Image.open('examples/1.jpg')
crops = platerec.detect_read(image)
for idx, crop in enumerate(crops['pil']['images']):
    crop.save(f'{idx}.jpg')
```

pred will be something like:
```
{'images': [<PIL.Image.Image image mode=RGB size=105x40 at 0x7FEE25B67AD0>], 'confidences': array([0.72949219]), 'words': ['[br]AAA1A11'], 'boxes': array([[ 393, 1188,  498, 1228]], dtype=int32), 'words_confidences': [0.95263671875]}
```

If you want to use CUDA:
```python
from platerec import Platerec

platerec = Platerec(providers=["CUDAExecutionProvider"])
```

Check all execution providers [here](https://onnxruntime.ai/docs/execution-providers/).

<hr>

Extra commands for quick testing:

```bash
platerec-video video_path [--font_size FONT_SIZE] [--save_output]
```
Run platerec on a video file.

```bash
platerec-image image_path
```
Run platerec on a image file.