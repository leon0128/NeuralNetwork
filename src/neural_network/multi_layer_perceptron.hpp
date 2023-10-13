#ifndef NEURAL_NETWORK_MULTI_LAYER_PERCEPTRON_HPP
#define NEURAL_NETWORK_MULTI_LAYER_PERCEPTRON_HPP

#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <cmath>
#include <deque>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <limits>

#include "matrix/matrix.hpp"
#include "random.hpp"
#include "layer.hpp"
#include "weight.hpp"
#include "bias.hpp"
#include "function.hpp"
#include "tag.hpp"

template<class T>
class MultiLayerPerceptron
{
public:
    MultiLayerPerceptron();
    ~MultiLayerPerceptron();

    void addLayer(Layer<T> *layer);
    
    bool train(std::size_t epochSize
        , std::size_t batchSize
        , ErrorTag errorTag
        , OptimizationTag optimizationTag
        , const std::vector<Matrix<T>> &trainingInput
        , const std::vector<Matrix<T>> &trainingOutput
        , const std::vector<Matrix<T>> &validationInput
        , const std::vector<Matrix<T>> &validationOutput
        , const std::vector<Matrix<T>> &testInput
        , const std::vector<Matrix<T>> &testOutput
        , bool shouldStopEarly = true);

    bool activate(const Matrix<T> &input
        , Matrix<T> &output);

private:
    bool checkValidity(std::size_t epochSize
        , std::size_t batchSize
        , ErrorTag errorTag
        , OptimizationTag optimizationTag
        , const std::vector<Matrix<T>> &trainingInput
        , const std::vector<Matrix<T>> &trainingOutput
        , const std::vector<Matrix<T>> &validationInput
        , const std::vector<Matrix<T>> &validationOutput
        , const std::vector<Matrix<T>> &testInput
        , const std::vector<Matrix<T>> &testOutput) const;
    bool checkValidity(const Matrix<T> &input) const;
    bool randomizeParameter();
    bool propagate(const Matrix<T> &trainingInput);
    bool backpropagate(const Matrix<T> &trainingOutput
        , ErrorTag errorTag
        , std::list<Weight<T>*> &weightGradients
        , std::list<Bias<T>*> &biasGradients);
    bool calculateAverage(std::size_t batchSize
        , std::list<Weight<T>*> &weightGradients
        , std::list<Bias<T>*> &biasGradients);
    bool updateParameter(OptimizationTag optimizationTag
        , std::list<Weight<T>*> &weightGradients
        , std::list<Bias<T>*> &biasGradients);
    T calculateError(const std::vector<Matrix<T>> &inputs
        , const std::vector<Matrix<T>> &outputs
        , ErrorTag errorTag);

    bool trainingError(const std::string &what) const;
    bool activationError(const std::string &what) const;

    std::list<Layer<T>*> mLayers;
    std::list<Weight<T>*> mWeights;
    std::list<Bias<T>*> mBiases;
};

template<class T>
MultiLayerPerceptron<T>::MultiLayerPerceptron()
    : mLayers{}
    , mWeights{}
    , mBiases{}
{
}

template<class T>
MultiLayerPerceptron<T>::~MultiLayerPerceptron()
{
    for(auto &&layer : mLayers)
        delete layer;
    for(auto &&weight : mWeights)
        delete weight;
    for(auto &&bias : mBiases)
        delete bias;
}

template<class T>
void MultiLayerPerceptron<T>::addLayer(Layer<T> *layer)
{
    // if layer is not input layer,
    //  weight and bias is created.
    if(!mLayers.empty())
    {
        auto &&lastLayer{mLayers.back()};
        Weight<T> *weight{new Weight<T>{lastLayer->output().column()
            , layer->input().column()}};
        Bias<T> *bias{new Bias<T>{layer->input().column()}};
        mWeights.push_back(weight);
        mBiases.push_back(bias);
    }

    mLayers.push_back(layer);
}

template<class T>
bool MultiLayerPerceptron<T>::train(std::size_t epochSize
    , std::size_t batchSize
    , ErrorTag errorTag
    , OptimizationTag optimizationTag
    , const std::vector<Matrix<T>> &trainingInput
    , const std::vector<Matrix<T>> &trainingOutput
    , const std::vector<Matrix<T>> &validationInput
    , const std::vector<Matrix<T>> &validationOutput
    , const std::vector<Matrix<T>> &testInput
    , const std::vector<Matrix<T>> &testOutput
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

    T prevError{std::numeric_limits<T>::max()};
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
            std::list<Weight<T>*> weightGradients;
            std::list<Bias<T>*> biasGradients;
            for(auto &&weight : mWeights)
                weightGradients.push_back(new Weight<T>{weight->data().row(), weight->data().column()});
            for(auto &&bias : mBiases)
                biasGradients.push_back(new Bias<T>{bias->data().column()});

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

        auto &&error{calculateError(validationInput, validationOutput, errorTag)};
        
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

template<class T>
bool MultiLayerPerceptron<T>::activate(const Matrix<T> &input
    , Matrix<T> &output)
{
    if(!checkValidity(input))
        return false;

    if(!propagate(input))
        return false;
    
    output = mLayers.back()->output();
    return true;
}

template<class T>
bool MultiLayerPerceptron<T>::checkValidity(std::size_t epochSize
    , std::size_t batchSize
    , ErrorTag errorTag
    , OptimizationTag optimizationTag
    , const std::vector<Matrix<T>> &trainingInput
    , const std::vector<Matrix<T>> &trainingOutput
    , const std::vector<Matrix<T>> &validationInput
    , const std::vector<Matrix<T>> &validationOutput
    , const std::vector<Matrix<T>> &testInput
    , const std::vector<Matrix<T>> &testOutput) const
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

template<class T>
bool MultiLayerPerceptron<T>::checkValidity(const Matrix<T> &input) const
{
    if(mLayers.empty())
        return activationError("multi-layer perceptron has no layers");

    return true;
}

template<class T>
bool MultiLayerPerceptron<T>::randomizeParameter()
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
                std::normal_distribution<T> dist{0.0, std::sqrt(2.0 / (*prevLayerIter)->output().column())};
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

template<class T>
bool MultiLayerPerceptron<T>::propagate(const Matrix<T> &trainingInput)
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

template<class T>
bool MultiLayerPerceptron<T>::backpropagate(const Matrix<T> &trainingOutput
    , ErrorTag errorTag
    , std::list<Weight<T>*> &weightGradients
    , std::list<Bias<T>*> &biasGradients)
{
    // reverse iterators
    auto &&prevLayerIter{mLayers.rbegin()};
    prevLayerIter++;
    auto &&nextLayerIter{mLayers.rbegin()};
    auto &&weightIter{mWeights.rbegin()};
    auto &&biasIter{mBiases.rbegin()};
    auto &&weightGradientIter{weightGradients.rbegin()};
    auto &&biasGradientIter{biasGradients.rbegin()};

    auto &&derivativeErrorFunction{FUNCTION::derivativeErrorFunction<T>(errorTag)};
    auto &&derivativeActivationFunction{FUNCTION::derivativeActivationFunction<T>((*nextLayerIter)->activationTag())};

    // output layer
    Matrix<T> error{1ull, trainingOutput.column()};
    Matrix<T> dError{derivativeErrorFunction(trainingOutput, (*nextLayerIter)->output())};
    Matrix<T> dActivation{derivativeActivationFunction((*nextLayerIter)->output())};
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
        derivativeActivationFunction = FUNCTION::derivativeActivationFunction<T>((*nextLayerIter)->activationTag());
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

template<class T>
bool MultiLayerPerceptron<T>::calculateAverage(std::size_t batchSize
    , std::list<Weight<T>*> &weightGradients
    , std::list<Bias<T>*> &biasGradients)
{
    T denominator{static_cast<T>(batchSize)};

    for(auto &&gradient : weightGradients)
        gradient->data().apply([&](T in){return in / denominator;});
    for(auto &&gradient : biasGradients)
        gradient->data().apply([&](T in){return in / denominator;});

    return true;
}

template<class T>
bool MultiLayerPerceptron<T>::updateParameter(OptimizationTag optimizationTag
    , std::list<Weight<T>*> &weightGradients
    , std::list<Bias<T>*> &biasGradients)
{
    static std::list<Matrix<T>> weightAdamMs{};
    static std::list<Matrix<T>> biasAdamMs{};
    static std::list<Matrix<T>> weightAdamVs{};
    static std::list<Matrix<T>> biasAdamVs{};
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


    auto &&optimizationFunction{FUNCTION::optimizationFunction<T>(optimizationTag)};

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

template<class T>
T MultiLayerPerceptron<T>::calculateError(const std::vector<Matrix<T>> &inputs
    , const std::vector<Matrix<T>> &outputs
    , ErrorTag errorTag)
{
    auto &&errorFunction{FUNCTION::errorFunction<T>(errorTag)};

    T error{0};

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

template<class T>
bool MultiLayerPerceptron<T>::trainingError(const std::string &what) const
{
    std::cerr << "Mlp::trainingError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}

template<class T>
bool MultiLayerPerceptron<T>::activationError(const std::string &what) const
{
    std::cerr << "Mlp::activationError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}

#endif