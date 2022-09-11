/**
 * @file convert_util.hpp
 * @author Shubham Agrawal
 *
 * Utility functions for conerting weights from PyTorch
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef CONVERT_UTIL_HPP
#define CONVERT_UTIL_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/make_alias.hpp>
#include <rapidxml/rapidxml_utils.hpp>

// Save a mlpack object.
template<typename T, typename IArchiveType>
void SerializeObject(T& t, std::string fileName)
{
  std::ofstream ofs(fileName, std::ios::binary);

  {
    OArchiveType o(ofs);

    T& x(t);
    o(CEREAL_NVP(x));
  }
  ofs.close();
}

/**
 * Processes and initializes BatchNorm layers with correct running parameters.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 * @param network ANN Network in which we want to initialize running parameters.
 * @param runningParams Queue containing matrix of running Parameters.
 */
template<typename MatType = arma::mat>
void ProcessBatchNorm(std::vector<mlpack::ann::Layer<MatType>*>& network, std::queue<arma::mat>* runningParams)
{
  if (runningParams->empty())
    return;

  for (size_t i = 0; i < network.size(); i++)
  {
    mlpack::ann::BatchNormType<MatType>* batchNorm = dynamic_cast<mlpack::ann::BatchNormType<MatType>*>(network[i]);
    if (batchNorm != nullptr)
    {
      batchNorm->TrainingMean() = runningParams->front().t();
      runningParams->pop();
      batchNorm->TrainingVariance() = runningParams->front().t();
      runningParams->pop();
      continue;
    }
    mlpack::ann::MultiLayer<MatType>* multiLayer = dynamic_cast<mlpack::ann::MultiLayer<MatType>*>(network[i]);
    if (multiLayer != nullptr)
    {
      ProcessBatchNorm(multiLayer->Network(), runningParams);
    }
  }
}

/**
 * Processes XML file and converts it to sequential parameter matrix.
 *
 * @param rootNode The layer node which we want to process
 * @param runningParams Queue containing matrix of running Parameters.
 */
size_t ProcessXML(rapidxml::xml_node<>* rootNode, double* paramsPtr, std::queue<arma::mat>* runningParams)
{
  size_t start = 0;
  for (rapidxml::xml_node<>* node = rootNode->first_node("layer"); node; node = node->next_sibling("layer"))
  {
    std::string layerName = node->first_attribute()->value();
    std::string isLeaf = node->first_node("is_leaf")->value();
    if (isLeaf == "0")
    {
      start += ProcessXML(node, paramsPtr + start, runningParams);
    }
    else
    {
      std::string hasWeight = node->first_node("has_weight")->value();
      std::string hasBias = node->first_node("has_bias")->value();
      if (hasWeight == "1")
      {
        arma::mat weight;
        mlpack::data::Load(node->first_node("weight_csv")->value(), weight);
        arma::mat wTemp;
        mlpack::ann::MakeAlias(wTemp, paramsPtr + start, weight.n_elem, 1);
        wTemp = weight.t();
        start += weight.n_elem;
      }
      if (hasBias == "1")
      {
        arma::mat bias;
        mlpack::data::Load(node->first_node("bias_csv")->value(), bias);
        arma::mat bTemp;
        mlpack::ann::MakeAlias(bTemp, paramsPtr + start, bias.n_elem, 1);
        bTemp = bias.t();
        start += bias.n_elem;
      }
      if (layerName == "BatchNorm2d")
      {
        arma::mat runningMean;
        mlpack::data::Load(node->first_node("running_mean_csv")->value(), runningMean);
        runningParams->push(runningMean);
        arma::mat runningVar;
        mlpack::data::Load(node->first_node("running_var_csv")->value(), runningVar);
        runningParams->push(runningVar);
      }
    }
  }
  return start;
}

/**
 * Reads XML file and converts it to sequential parameter matrix.
 *
 * @param filename The filename of xml file.
 * @param runningParams Queue containing matrix of running Parameters.
 */
arma::mat ReadXML(std::string filename, std::queue<arma::mat>* runningParams)
{
  rapidxml::file<> xmlFile(&filename[0]);
  rapidxml::xml_document<> doc;
  doc.parse<0>(xmlFile.data());

  rapidxml::xml_node<>* rootNode = doc.first_node();
  std::string numParams = rootNode->first_node("trainable_param")->value();
  std::stringstream sstream(numParams);
  size_t numParamsInt;
  sstream >> numParamsInt;

  arma::mat params(numParamsInt, 1);
  ProcessXML(rootNode, params.memptr(), runningParams);
  return params;
}

#endif // CONVERT_UTIL_HPP
