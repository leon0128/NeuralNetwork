#include <iostream>
#include <cmath>
#include <deque>
#include <numeric>
#include <algorithm>
#include <stdexcept>

#include "random.hpp"
#include "function.hpp"
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
    , OptimizationTag optimizationTag
    , const std::vector<Matrix<double>> &trainingInput
    , const std::vector<Matrix<double>> &trainingOutput
    , const std::vector<Matrix<double>> &validationInput
    , const std::vector<Matrix<double>> &validationOutput
    , const std::vector<Matrix<double>> &testInput
    , const std::vector<Matrix<double>> &testOutput)
{
    if(!checkValidity(epochSize
        , batchSize
        , errorTag
        , optimizationTag
        , trainingInput
        , trainingOutput
        , validationInput
        , validationOutput
        , testInput
        , testOutput))
        return false;

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
            std::list<Weight*> weightGradients;
            std::list<Bias*> biasGradients;
            for(auto &&weight : mWeights)
                weightGradients.push_back(new Weight{weight->data().row(), weight->data().column()});
            for(auto &&bias : mBiases)
                biasGradients.push_back(new Bias{bias->data().column()});

            for(std::size_t batch{0ull}; batch < batchSize; batch++)
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
                    return false;
                if(!backpropagate(trainingOutput[trainingIndex]
                    , errorTag
                    , weightGradients
                    , biasGradients))
                    return false;
            }

            if(!calculateAverage(batchSize
                , weightGradients
                , biasGradients))
                return false;
            if(!updateParameter(optimizationTag
                , weightGradients
                , biasGradients))
                return false;

            for(auto &&gradient : weightGradients)
                delete gradient;
            for(auto &&gradient : biasGradients)
                delete gradient;
        }
    }

    return true;
}

bool Mlp::checkValidity(std::size_t epochSize
    , std::size_t batchSize
    , ErrorTag errorTag
    , OptimizationTag optimizationTag
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
    if(epochSize == 0ull
        || batchSize == 0ull)
        return trainingError("epoch/batch sizes is invalid.");
    if(errorTag == ErrorTag::NONE
        || optimizationTag == OptimizationTag::NONE)
        return trainingError("error/optimization should be selected.");
    if(mLayers.empty())
        return trainingError("multi-layer perceptron has no layers.");

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
        switch((*layerIter)->activationTag())
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

bool Mlp::propagate(const Matrix<double> &trainingInput)
{
    auto &&prevLayerIter{mLayers.begin()};
    auto &&nextLayerIter{mLayers.begin()};
    nextLayerIter++;
    auto &&weightIter{mWeights.begin()};
    auto &&biasIter{mBiases.begin()};

    // input layer
    (*prevLayerIter)->input(trainingInput);
    if(!(*prevLayerIter)->activate())
        return false;
    
    // others
    for(; nextLayerIter != mLayers.end(); prevLayerIter++, nextLayerIter++, weightIter++, biasIter++)
    {
        (*nextLayerIter)->input((*prevLayerIter)->input() * (*weightIter)->data());
        (*nextLayerIter)->input() += (*biasIter)->data();
        if(!(*nextLayerIter)->activate())
            return false;
    }

    return true;
}

bool Mlp::backpropagate(const Matrix<double> &trainingOutput
    , ErrorTag errorTag
    , std::list<Weight*> &weightGradients
    , std::list<Bias*> &biasGradients)
{
    // reverse iterators
    auto &&prevLayerIter{mLayers.rbegin()};
    prevLayerIter++;
    auto &&nextLayerIter{mLayers.rbegin()};
    auto &&weightIter{mWeights.rbegin()};
    auto &&biasIter{mBiases.rbegin()};
    auto &&weightGradientIter{weightGradients.rbegin()};
    auto &&biasGradientIter{biasGradients.rbegin()};

    auto &&differentiatedErrorFunction{FUNCTION::differentiatedErrorFunction<double>(errorTag)};
    auto &&differentiatedActivationFunction{FUNCTION::differentiatedActivationFunction<double>((*nextLayerIter)->activationTag())};

    // output layer
    Matrix<double> error{1ull, (*nextLayerIter)->output().column()};
    for(std::size_t c{0ull}; c < error.column(); c++)
    {
        error[0ull][c]
            = differentiatedErrorFunction(trainingOutput[0ull][c], (*nextLayerIter)->output()[0ull][c])
                * differentiatedActivationFunction((*nextLayerIter)->output()[0ull][c]);
    }
    (*weightGradientIter)->data() += ~(*prevLayerIter)->output() * error;
    (*biasGradientIter)->data() += error;

    // others
    for(prevLayerIter++
            , nextLayerIter++
            , weightGradientIter++
            , biasGradientIter++;
        prevLayerIter != mLayers.rend();
        prevLayerIter++
            , nextLayerIter++
            , weightIter++
            , biasIter++
            , weightGradientIter++
            , biasGradientIter++)
    {
        differentiatedActivationFunction = FUNCTION::differentiatedActivationFunction<double>((*nextLayerIter)->activationTag());
        error = error * ~(*weightIter)->data();
        for(std::size_t c{0ull}; c < error.column(); c++)
            error[0ull][c] *= differentiatedActivationFunction((*nextLayerIter)->output()[0ull][c]);
        (*weightGradientIter)->data() += ~(*prevLayerIter)->output() * error;
        (*biasGradientIter)->data() += error;
    }

    return true;
}

bool Mlp::calculateAverage(std::size_t batchSize
    , std::list<Weight*> &weightGradients
    , std::list<Bias*> &biasGradients)
{
    double denominator{static_cast<double>(batchSize)};

    for(auto &&gradient : weightGradients)
        gradient->data().apply([&](double in){return in / denominator;});
    for(auto &&gradient : biasGradients)
        gradient->data().apply([&](double in){return in / denominator;});

    return true;
}

bool Mlp::updateParameter(OptimizationTag optimizationTag
    , std::list<Weight*> &weightGradients
    , std::list<Bias*> &biasGradients)
{
    auto &&optimizationFunction{FUNCTION::optimizationFunction<double>(optimizationTag)};

    auto &&weightIter{mWeights.begin()};
    auto &&biasIter{mBiases.begin()};
    auto &&weightGradientIter{weightGradients.begin()};
    auto &&biasGradientIter{biasGradients.begin()};

    for(;
        weightIter != mWeights.end();
        weightIter++
            , biasIter++
            , weightGradientIter++
            , biasGradientIter++)
    {
        for(std::size_t r{0ull}; r < (*weightIter)->data().row(); r++)
        {
            for(std::size_t c{0ull}; c < (*weightIter)->data().column(); c++)
            {
                (*weightIter)->data()[r][c]
                    = optimizationFunction((*weightIter)->data()[r][c]
                        , (*weightGradientIter)->data()[r][c]);
            }
        }
        for(std::size_t c{0ull}; c < (*biasIter)->data().column(); c++)
        {
            (*biasIter)->data()[0ull][c]
                = optimizationFunction((*biasIter)->data()[0ull][c]
                    , (*biasGradientIter)->data()[0ull][c]);
        }
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