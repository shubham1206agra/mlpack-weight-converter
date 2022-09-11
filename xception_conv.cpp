/**
 * @file xception_conv.cpp
 * @author Shubham Agrawal
 *
 * Serializes the Xception model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "convert_util.hpp"
#include "../models/xception/xception.hpp"

using namespace mlpack;
using namespace mlpack::ann;

int main()
{
  std::queue<arma::mat> *runningParams = new std::queue<arma::mat>();

  FFN<> net;
  net.Add<models::Xception>();
  net.InputDimensions() = std::vector<size_t>({224, 224, 3});

  net.Parameters() = ReadXML("./xception.xml", runningParams);
  ProcessBatchNorm(net.Network(), runningParams);

  arma::mat input(224 * 224 * 3, 2, arma::fill::randu);
  arma::mat output;

  net.Predict(input, output);

  SerializeObject<FFN<>, cereal::BinaryInputArchive>(net, "xception.bin");
}