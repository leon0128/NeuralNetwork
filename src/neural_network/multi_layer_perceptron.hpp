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
#include <memory>

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
        , const std::vector<Matrix<T>> &testOutput
        , bool shouldStopEarly) const;
    bool checkValidity(const Matrix<T> &input) const;
    bool randomizeParameter();
    std::deque<std::size_t> createTrainingIndices(std::size_t trainingSize
        , std::size_t batchSize) const;
    bool propagate(const Matrix<T> &trainingInput);
    bool backpropagate(const Matrix<T> &trainingOutput
        , ErrorTag errorTag
        , std::list<std::shared_ptr<Weight<T>>> &weightGradients
        , std::list<std::shared_ptr<Bias<T>>> &biasGradients);
    bool calculateAverage(std::size_t batchSize
        , std::list<std::shared_ptr<Weight<T>>> &weightGradients
        , std::list<std::shared_ptr<Bias<T>>> &biasGradients);
    bool updateParameter(OptimizationTag optimizationTag
        , std::list<std::shared_ptr<Weight<T>>> &weightGradients
        , std::list<std::shared_ptr<Bias<T>>> &biasGradients);
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
        , testOutput
        , shouldStopEarly))
        return false;

    if(!randomizeParameter())
        return false;

    T prevError{std::numeric_limits<T>::max()};
    std::list<std::shared_ptr<Weight<T>>> weightGradients;
    std::list<std::shared_ptr<Bias<T>>> biasGradients;
    for(auto &&weight : mWeights)
        weightGradients.emplace_back(new Weight<T>{weight->data().row(), weight->data().column()});
    for(auto &&bias : mBiases)
        biasGradients.emplace_back(new Bias<T>{bias->data().column()});

    for(std::size_t epoch{0ull}; epoch < epochSize; epoch++)
    {
        for(auto &&trainingIndices{createTrainingIndices(trainingInput.size(), batchSize)};
            !trainingIndices.empty();
            trainingIndices.pop_front())
        {
            std::size_t trainingIndex{trainingIndices.front()};
            if(!propagate(trainingInput[trainingIndex]))
                return false;
            if(!backpropagate(trainingOutput[trainingIndex]
                , errorTag
                , weightGradients
                , biasGradients))
                return false;
            
            if((trainingIndices.size() - 1ull) % batchSize != 0ull)
                continue;
            
            if(!calculateAverage(batchSize
                , weightGradients
                , biasGradients))
                return false;
            if(!updateParameter(optimizationTag
                , weightGradients
                , biasGradients))
                return false;
            
            for(auto &&gradient : weightGradients)
                gradient->data().apply([](T in){return T{0};});
            for(auto &&gradient : biasGradients)
                gradient->data().apply([](T in){return T{0};});
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
    , const std::vector<Matrix<T>> &testOutput
    , bool shouldStopEarly) const
{
    auto &&checkMatrixValidity{[](auto &&vec, auto &&columnSize)
        -> bool
        {
            for(auto &&matrix : vec)
            {
                if(matrix.row() != 1ull
                    || matrix.column() != columnSize)
                    return false;
            }
            return true;
        }};

    if(trainingInput.size() != trainingOutput.size()
        || validationInput.size() != validationOutput.size()
        || testInput.size() != testOutput.size()
        || trainingInput.empty()
        || trainingInput.empty()
        || validationInput.empty())
        return trainingError("data size for training is invalid.");
    if(epochSize == 0ull
        || batchSize == 0ull)
        return trainingError("epoch/batch size is invalid.");
    if(errorTag == ErrorTag::NONE)
        return trainingError("error should be selected.");
    if(mLayers.empty())
        return trainingError("multi-layer perceptron has no layers.");

    if(!checkMatrixValidity(trainingInput, mLayers.front()->input().column())
        || !checkMatrixValidity(trainingOutput, mLayers.back()->output().column())
        || !checkMatrixValidity(validationInput, mLayers.front()->input().column())
        || !checkMatrixValidity(validationOutput, mLayers.back()->output().column())
        || !checkMatrixValidity(testInput, mLayers.front()->input().column())
        || !checkMatrixValidity(testOutput, mLayers.back()->output().column()))
        return trainingError("data's elements does not match layer's io.");

    for(auto &&layer : mLayers)
        if(layer->input().column() == 0ull)
            return trainingError("layer has 0 size's input");

    return true;
}

template<class T>
bool MultiLayerPerceptron<T>::checkValidity(const Matrix<T> &input) const
{
    if(mLayers.empty())
        return activationError("multi-layer perceptron has no layers");
    for(auto &&layer : mLayers)
        if(layer->input().column() == 0ull)
            return activationError("layer has 0 size's input.");

    if(input.row() != 1ull
        || input.column() != mLayers.front()->input().column())
        return activationError("input size does not match layer's input.");

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
std::deque<std::size_t> MultiLayerPerceptron<T>::createTrainingIndices(std::size_t trainingSize
    , std::size_t batchSize) const
{
    std::deque<std::size_t> indices(trainingSize);
    std::iota(indices.begin()
        , indices.end()
        , 0ull);
    std::shuffle(indices.begin()
        , indices.end()
        , RANDOM.engine());

    for(std::size_t i{0ull}, size{(batchSize - trainingSize % batchSize) % batchSize}; i < size; i++)
        indices.push_back(RANDOM(trainingSize));

    return indices;
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
    , std::list<std::shared_ptr<Weight<T>>> &weightGradients
    , std::list<std::shared_ptr<Bias<T>>> &biasGradients)
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
    , std::list<std::shared_ptr<Weight<T>>> &weightGradients
    , std::list<std::shared_ptr<Bias<T>>> &biasGradients)
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
    , std::list<std::shared_ptr<Weight<T>>> &weightGradients
    , std::list<std::shared_ptr<Bias<T>>> &biasGradients)
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
    std::cerr << "MultiLayerPerceptron::trainingError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}

template<class T>
bool MultiLayerPerceptron<T>::activationError(const std::string &what) const
{
    std::cerr << "MultiLayerPerceptron::activationError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}

#endif