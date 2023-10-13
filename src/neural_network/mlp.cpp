#include <iostream>
#include <cmath>
#include <deque>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <limits>

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
    , const std::vector<Matrix<double>> &testOutput
    , bool shouldStopEarly)
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

    double prevError{std::numeric_limits<double>::max()};
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

        double error{calculateError(validationInput, validationOutput, errorTag)};
        
        if(epochSize < 10 || (epoch + 1ull) % (epochSize / 10ull) == 0)
            std::cout << "epoch " << epoch + 1ull << "'s error: " << error << std::endl;
        if(shouldStopEarly && error > prevError)
        {
            std::cout << "early stopping has been activated."
                << "\n    reached epoch: " << epoch + 1ull << "/" << epochSize << std::endl;
            break;
        }

        prevError = error;
    }

    return true;
}

bool Mlp::activate(const Matrix<double> &input
    , Matrix<double> &output)
{
    if(!checkValidity(input))
        return false;

    if(!propagate(input))
        return false;
    
    output = mLayers.back()->output();
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
    , const std::vector<Matrix<double>> &testOutput) const
{
    if(trainingInput.size() != trainingOutput.size()
        || validationInput.size() != validationOutput.size()
        || testInput.size() != testOutput.size()
        || trainingInput.empty())
        return trainingError("data sizes for training is invalid.");
    if(epochSize == 0ull
        || batchSize == 0ull)
        return trainingError("epoch/batch sizes is invalid.");
    if(errorTag == ErrorTag::NONE)
        return trainingError("error should be selected.");
    if(mLayers.empty())
        return trainingError("multi-layer perceptron has no layers.");

    return true;
}

bool Mlp::checkValidity(const Matrix<double> &input) const
{
    if(mLayers.empty())
        return activationError("multi-layer perceptron has no layers");

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
            case(ActivationTag::NONE):
            case(ActivationTag::SIGMOID):
            case(ActivationTag::ELU):
            case(ActivationTag::SOFTMAX):
            case(ActivationTag::RELU):
            {
                std::normal_distribution<> dist{0.0, std::sqrt(2.0 / (*prevLayerIter)->output().column())};
                for(std::size_t r{0ull}; r < (*weightIter)->data().row(); r++)
                    for(std::size_t c{0ull}; c < (*weightIter)->data().column(); c++)
                        (*weightIter)->data()[r][c] = dist(RANDOM.engine());
                for(std::size_t c{0ull}; c < (*biasIter)->data().column(); c++)
                    (*biasIter)->data()[0ull][c] = 0.1;

                break;
            }
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
        (*nextLayerIter)->input((*prevLayerIter)->output() * (*weightIter)->data());
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

    auto &&derivativeErrorFunction{FUNCTION::derivativeErrorFunction<double>(errorTag)};
    auto &&derivativeActivationFunction{FUNCTION::derivativeActivationFunction<double>((*nextLayerIter)->activationTag())};

    // output layer
    Matrix<double> error{1ull, trainingOutput.column()};
    Matrix<double> dError{derivativeErrorFunction(trainingOutput, (*nextLayerIter)->output())};
    Matrix<double> dActivation{derivativeActivationFunction((*nextLayerIter)->output())};
    switch((*nextLayerIter)->activationTag())
    {
        case(ActivationTag::NONE):
        case(ActivationTag::ELU):
        case(ActivationTag::SIGMOID):
        case(ActivationTag::RELU):
            for(std::size_t c{0ull}; c < error.column(); c++)
                error[0ull][c] = dError[0ull][c] * dActivation[0ull][c];
            break;
        case(ActivationTag::SOFTMAX):
            error = dError * dActivation;
            break;
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
        derivativeActivationFunction = FUNCTION::derivativeActivationFunction<double>((*nextLayerIter)->activationTag());
        error = error * ~(*weightIter)->data();
        dActivation = derivativeActivationFunction((*nextLayerIter)->output());
        switch((*nextLayerIter)->activationTag())
        {
            case(ActivationTag::ELU):
            case(ActivationTag::NONE):
            case(ActivationTag::RELU):
            case(ActivationTag::SIGMOID):
                for(std::size_t c{0ull}; c < error.column(); c++)
                    error[0ull][c] *= dActivation[0ull][c];
                break;
            case(ActivationTag::SOFTMAX):
                error = error * dActivation;
        }

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
    static std::list<Matrix<double>> weightAdamMs{};
    static std::list<Matrix<double>> biasAdamMs{};
    static std::list<Matrix<double>> weightAdamVs{};
    static std::list<Matrix<double>> biasAdamVs{};
    static bool isInitialized{false};
    if(!isInitialized)
    {
        for(auto &&gradient : weightGradients)
        {
            weightAdamMs.emplace_back(gradient->data().row()
                , gradient->data().column());
            weightAdamVs.emplace_back(gradient->data().row()
                , gradient->data().column());
        }
        for(auto &&gradient : biasGradients)
        {
            biasAdamMs.emplace_back(gradient->data().row()
                , gradient->data().column());
            biasAdamVs.emplace_back(gradient->data().row()
                , gradient->data().column());
        }
        isInitialized = true;
    }


    auto &&optimizationFunction{FUNCTION::optimizationFunction<double>(optimizationTag)};

    auto &&weightIter{mWeights.begin()};
    auto &&biasIter{mBiases.begin()};
    auto &&weightGradientIter{weightGradients.begin()};
    auto &&biasGradientIter{biasGradients.begin()};
    auto &&weightAdamMIter{weightAdamMs.begin()};
    auto &&weightAdamVIter{weightAdamVs.begin()};
    auto &&biasAdamMIter{biasAdamMs.begin()};
    auto &&biasAdamVIter{biasAdamVs.begin()};

    for(;
        weightIter != mWeights.end();
        weightIter++
            , biasIter++
            , weightGradientIter++
            , biasGradientIter++
            , weightAdamMIter++
            , weightAdamVIter++
            , biasAdamMIter++
            , biasAdamVIter++)
    {
        switch(optimizationTag)
        {
            case(OptimizationTag::NONE):
                break;
            case(OptimizationTag::ADAM):
                FUNCTION::adamM = *weightAdamMIter;
                FUNCTION::adamV = *weightAdamVIter;
                break;
        }

        (*weightIter)->data()
            = optimizationFunction((*weightIter)->data()
                , (*weightGradientIter)->data());

        switch(optimizationTag)
        {
            case(OptimizationTag::NONE):
                break;
            case(OptimizationTag::ADAM):
                *weightAdamMIter = FUNCTION::adamM;
                *weightAdamVIter = FUNCTION::adamV;
                FUNCTION::adamM = *biasAdamMIter;
                FUNCTION::adamV = *biasAdamVIter;
                break;
        }

        (*biasIter)->data()
            = optimizationFunction((*biasIter)->data()
                , (*biasGradientIter)->data());

        switch(optimizationTag)
        {
            case(OptimizationTag::NONE):
                break;
            case(OptimizationTag::ADAM):
                *biasAdamMIter = FUNCTION::adamM;
                *biasAdamVIter = FUNCTION::adamV;
                break;
        }
    }

    return true;
}

double Mlp::calculateError(const std::vector<Matrix<double>> &inputs
    , const std::vector<Matrix<double>> &outputs
    , ErrorTag errorTag)
{
    auto &&errorFunction{FUNCTION::errorFunction<double>(errorTag)};

    double error{0.0};

    for(auto &&inputIter{inputs.begin()}
            , &&outputIter{outputs.begin()};
        inputIter != inputs.end();
        inputIter++
            , outputIter++)
    {
        propagate(*inputIter);
        error += errorFunction((*outputIter), mLayers.back()->output());
    }

    return error;
}

bool Mlp::trainingError(const std::string &what) const
{
    std::cerr << "Mlp::trainingError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}

bool Mlp::activationError(const std::string &what) const
{
    std::cerr << "Mlp::activationError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}