/**
 * @file vgg_conv.cpp
 * @author Shubham Agrawal
 *
 * Serializes the VGG models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "convert_util.hpp"
#include "../models/vgg/vgg.hpp"

using namespace mlpack;
using namespace mlpack::ann;

int main()
{
  std::queue<arma::mat> *runningParams = new std::queue<arma::mat>();
  FFN<> net;
  net.Add<models::VGG11>();
  net.InputDimensions() = std::vector<size_t>({224, 224, 3});
  net.Parameters() = ReadXML("./vgg11.xml", runningParams);
  ProcessBatchNorm(net.Network(), runningParams);
  arma::mat input(224 * 224 * 3, 2, arma::fill::randu);
  arma::mat output;
  net.Predict(input, output);
  SerializeObject<FFN<>, cereal::BinaryInputArchive>(net, "vgg11.bin");
  
  runningParams = new std::queue<arma::mat>();
  net = FFN<>();
  net.Add<models::VGG13>();
  net.InputDimensions() = std::vector<size_t>({224, 224, 3});
  net.Parameters() = ReadXML("./vgg13.xml", runningParams);
  ProcessBatchNorm(net.Network(), runningParams);
  net.Predict(input, output);
  SerializeObject<FFN<>, cereal::BinaryInputArchive>(net, "vgg13.bin");
  
  runningParams = new std::queue<arma::mat>();
  net = FFN<>();
  net.Add<models::VGG16>();
  net.InputDimensions() = std::vector<size_t>({224, 224, 3});
  net.Parameters() = ReadXML("./vgg16.xml", runningParams);
  ProcessBatchNorm(net.Network(), runningParams);
  net.Predict(input, output);
  SerializeObject<FFN<>, cereal::BinaryInputArchive>(net, "vgg16.bin");
  
  runningParams = new std::queue<arma::mat>();
  net = FFN<>();
  net.Add<models::VGG19>();
  net.InputDimensions() = std::vector<size_t>({224, 224, 3});
  net.Parameters() = ReadXML("./vgg19.xml", runningParams);
  ProcessBatchNorm(net.Network(), runningParams);
  net.Predict(input, output);
  SerializeObject<FFN<>, cereal::BinaryInputArchive>(net, "vgg19.bin");
  
  runningParams = new std::queue<arma::mat>();
  net = FFN<>();
  net.Add<models::VGG11BN>();
  net.InputDimensions() = std::vector<size_t>({224, 224, 3});
  net.Parameters() = ReadXML("./vgg11_bn.xml", runningParams);
  ProcessBatchNorm(net.Network(), runningParams);
  net.Predict(input, output);
  SerializeObject<FFN<>, cereal::BinaryInputArchive>(net, "vgg11_bn.bin");
  
  runningParams = new std::queue<arma::mat>();
  net = FFN<>();
  net.Add<models::VGG13BN>();
  net.InputDimensions() = std::vector<size_t>({224, 224, 3});
  net.Parameters() = ReadXML("./vgg13_bn.xml", runningParams);
  ProcessBatchNorm(net.Network(), runningParams);
  net.Predict(input, output);
  SerializeObject<FFN<>, cereal::BinaryInputArchive>(net, "vgg13_bn.bin");
  
  runningParams = new std::queue<arma::mat>();
  net = FFN<>();
  net.Add<models::VGG16BN>();
  net.InputDimensions() = std::vector<size_t>({224, 224, 3});
  net.Parameters() = ReadXML("./vgg16_bn.xml", runningParams);
  ProcessBatchNorm(net.Network(), runningParams);
  net.Predict(input, output);
  SerializeObject<FFN<>, cereal::BinaryInputArchive>(net, "vgg16_bn.bin");
  
  runningParams = new std::queue<arma::mat>();
  net = FFN<>();
  net.Add<models::VGG19BN>();
  net.InputDimensions() = std::vector<size_t>({224, 224, 3});
  net.Parameters() = ReadXML("./vgg19_bn.xml", runningParams);
  ProcessBatchNorm(net.Network(), runningParams);
  net.Predict(input, output);
  SerializeObject<FFN<>, cereal::BinaryInputArchive>(net, "vgg19_bn.bin");
}