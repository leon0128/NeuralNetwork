#include <iostream>
#include <cmath>
#include <deque>
#include <numeric>
#include <algorithm>

#include "random.hpp"
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

    if(!randomizeParameter())
        return false;

    for(std::size_t epoch{0ull}; epoch < epochSize; epoch++)
    {
        std::deque<std::size_t> trainingIndices(trainingInput.size());
        std::iota(trainingIndices.begin()
            , trainingIndices.end()
            , 0ull);
        std::shuffle(trainingIndices.begin()
            , trainingIndices.end()
            , RANDOM.engine());

        while(!trainingIndices.empty())
        {
            for(std::size_t batch; batch < batchSize; batch++)
            {
                std::size_t trainingIndex{0ull};
                if(!trainingIndices.empty())
                {
                    trainingIndex = trainingIndices.front();
                    trainingIndices.pop_front();
                }
                else
                    trainingIndex = RANDOM(trainingInput.size());

                if(!propagate(trainingInput[trainingIndex]))
            }
        }
    }

    return true;
}

bool Mlp::randomizeParameter()
{
    auto &&prevLayerIter{mLayers.begin()};
    auto &&layerIter{mLayers.begin()};
    layerIter++;
    auto &&weightIter{mWeights.begin()};
    auto &&biasIter{mBiases.begin()};

    while(layerIter != mLayers.end())
    {
        switch((*layerIter)->activation())
        {
            case(ActivationTag::ELU):
            {
                std::normal_distribution<> dist{0.0, std::sqrt(2.0 / (*prevLayerIter)->output().column())};
                for(std::size_t r{0ull}; r < (*weightIter)->data().row(); r++)
                    for(std::size_t c{0ull}; c < (*weightIter)->data().column(); c++)
                        (*weightIter)->data()[r][c] = dist(RANDOM.engine());
                for(std::size_t c{0ull}; c < (*biasIter)->data().column(); c++)
                    (*biasIter)->data()[0ull][c] = 0.1;
                break;
            }
            case(ActivationTag::NONE):
                break;
        }

        prevLayerIter++;
        layerIter++;
        weightIter++;
        biasIter++;
    }
    
    return true;
}

bool Mlp::trainingError(const std::string &what) const
{
    std::cerr << "Mlp::trainingError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}