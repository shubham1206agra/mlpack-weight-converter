
# mlpack-weight-converter

Simple Repository to transfer weights from PyTorch to mlpack.
#### Aim:

1.  Generate CSV files for weights, biases and all trainable parameters from PyTorch models.
2.  Generate XML file for PyTorch model that holds the structure as well as files necessary to reproduce the model.
3.  Create a parser in C++ which loads all weights and biases from XML file to the mlpack model.
4.  Test it on different models such as AlexNet, SqueezeNet, VGG, etc.

#### Status :
Complete

### Requirements :
1. Mlpack
2. Ensmallen
3. Armadillo
4. Cereal
5. RapidXML
6. Python 3.x
	1. torchinfo
	2. pytorch
	3. munch
	4. lxml
	5.  numpy

### Usage :

#### Basic Usage
For converting models given inside [shubham1206agra](https://github.com/shubham1206agra) / **[pretrained-models.pytorch](https://github.com/shubham1206agra/pretrained-models.pytorch)**, please use it as
```bash
python3 convert_weight.py --model "model_name"
```
After converting weights to CSV and XML file, parse it to mlpack model using `convert_util.hpp`. For example, see any `*_conv.cpp` file.

#### Advanced Usage
For converting any other models, pass the model inside `generate_xml` function inside `convert.py` file.
