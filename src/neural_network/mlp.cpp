#include <iostream>

#include "bias.hpp"
#include "weight.hpp"
#include "layer.hpp"
#include "mlp.hpp"

Mlp::Mlp()
    : mLayers{}
    , mWeights{}
    , mBiases{}
{
}

Mlp::~Mlp()
{
    for(auto &&layer : mLayers)
        delete layer;
    for(auto &&weight : mWeights)
        delete weight;
    for(auto &&bias : mBiases)
        delete bias;
}

void Mlp::addLayer(Layer *layer)
{
    // if layer is not input layer,
    //  weight and bias is created.
    if(!mLayers.empty())
    {
        auto &&lastLayer{mLayers.back()};
        Weight *weight{new Weight{lastLayer->output().column()
            , layer->input().column()}};
        Bias *bias{new Bias{layer->input().column()}};
        mWeights.push_back(weight);
        mBiases.push_back(bias);
    }

    mLayers.push_back(layer);
}

bool Mlp::train(std::size_t epochSize
    , std::size_t batchSize
    , ErrorTag errorTag
    , const std::vector<Matrix<double>> &trainingInput
    , const std::vector<Matrix<double>> &trainingOutput
    , const std::vector<Matrix<double>> &validationInput
    , const std::vector<Matrix<double>> &validationOutput
    , const std::vector<Matrix<double>> &testInput
    , const std::vector<Matrix<double>> &testOutput)
{
    if(trainingInput.size() != trainingOutput.size()
        || validationInput.size() != validationOutput.size()
        || testInput.size() != testOutput.size()
        || trainingInput.empty())
        return trainingError("data sizes for training is invalid.");

    return true;
}

bool Mlp::trainingError(const std::string &what) const
{
    std::cerr << "Mlp::trainingError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}