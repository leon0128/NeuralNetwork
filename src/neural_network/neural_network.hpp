#ifndef NEURAL_NETWORK_NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_NEURAL_NETWORK_HPP

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
#include <filesystem>
#include <fstream>
#include <iterator>

#include "matrix/matrix.hpp"
#include "concurrency.hpp"
#include "random.hpp"
#include "layer.hpp"
#include "weight.hpp"
#include "bias.hpp"
#include "function.hpp"
#include "tag.hpp"

namespace NEURAL_NETWORK
{

template<class T>
class NeuralNetwork
{
public:
    friend class Saver;
    friend class Loader;

    NeuralNetwork();
    ~NeuralNetwork();

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
        , std::size_t concurrency = 8ull
        , std::size_t earlyStoppingLimit = 5ull);

    bool predict(const Matrix<T> &input
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
        , std::size_t concurrency
        , std::size_t earlyStoppingLimit) const;
    bool checkValidity(const Matrix<T> &input) const;
    bool randomizeParameter();
    bool trainParameter(std::size_t epochSize
        , std::size_t batchSize
        , ErrorTag errorTag
        , OptimizationTag optimizationTag
        , const std::vector<Matrix<T>> &trainingInput
        , const std::vector<Matrix<T>> &trainingOutput
        , const std::vector<Matrix<T>> &validationInput
        , const std::vector<Matrix<T>> &validationOutput
        , std::size_t concurrency
        , std::size_t earlyStoppingLimit);
    std::deque<std::size_t> createTrainingIndices(std::size_t trainingSize
        , std::size_t batchSize) const;
    bool propagate(const Matrix<T> &trainingInput
        , bool isTraining);
    bool backpropagate(const Matrix<T> &trainingOutput
        , ErrorTag errorTag
        , std::list<std::shared_ptr<Weight<T>>> &weightGradients
        , std::list<std::shared_ptr<Bias<T>>> &biasGradients);
    bool calculateAverage(std::size_t batchSize
        , std::list<std::shared_ptr<Weight<T>>> &weightGradients
        , std::list<std::shared_ptr<Bias<T>>> &biasGradients);
    bool updateParameter(OptimizationTag optimizationTag
        , std::list<std::shared_ptr<Weight<T>>> &weightGradients
        , std::list<std::shared_ptr<Bias<T>>> &biasGradients
        , std::list<Matrix<T>> &weightAdamMs
        , std::list<Matrix<T>> &biasAdamMs
        , std::list<Matrix<T>> &weightAdamVs
        , std::list<Matrix<T>> &biasAdamVs);
    T calculateError(const std::vector<Matrix<T>> &inputs
        , const std::vector<Matrix<T>> &outputs
        , ErrorTag errorTag);
    bool shouldStopEarly(T error
        , T &minError
        , std::size_t earlyStoppingLimit
        , std::size_t &stoppingCount
        , std::list<std::shared_ptr<Weight<T>>> &minWeights
        , std::list<std::shared_ptr<Bias<T>>> &minBiases);

    bool trainingError(const std::string &what) const;
    bool activationError(const std::string &what) const;
    bool openingFileError(const std::filesystem::path &filepath) const;
    bool writingError(const std::string &what
        , const std::filesystem::path &filepath) const;
    bool readingError(const std::string &what
        , const std::filesystem::path &filepath) const;

    std::list<Layer<T>*> mLayers;
    std::list<Weight<T>*> mWeights;
    std::list<Bias<T>*> mBiases;
};

template<class T>
NeuralNetwork<T>::NeuralNetwork()
    : mLayers{}
    , mWeights{}
    , mBiases{}
{
}

template<class T>
NeuralNetwork<T>::~NeuralNetwork()
{
    for(auto &&layer : mLayers)
        delete layer;
    for(auto &&weight : mWeights)
        delete weight;
    for(auto &&bias : mBiases)
        delete bias;
}

template<class T>
void NeuralNetwork<T>::addLayer(Layer<T> *layer)
{
    // if layer is not input layer,
    //  weight and bias is created.
    if(!mLayers.empty())
    {
        auto &&lastLayer{mLayers.back()};
        Weight<T> *weight{new Weight<T>{lastLayer->data().column()
            , layer->data().column()}};
        Bias<T> *bias{new Bias<T>{layer->data().column()}};
        mWeights.push_back(weight);
        mBiases.push_back(bias);
    }

    mLayers.push_back(layer);
}

template<class T>
bool NeuralNetwork<T>::train(std::size_t epochSize
    , std::size_t batchSize
    , ErrorTag errorTag
    , OptimizationTag optimizationTag
    , const std::vector<Matrix<T>> &trainingInput
    , const std::vector<Matrix<T>> &trainingOutput
    , const std::vector<Matrix<T>> &validationInput
    , const std::vector<Matrix<T>> &validationOutput
    , const std::vector<Matrix<T>> &testInput
    , const std::vector<Matrix<T>> &testOutput
    , std::size_t concurrency
    , std::size_t earlyStoppingLimit)
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
        , concurrency
        , earlyStoppingLimit))
        return false;

    if(!randomizeParameter())
        return false;

    if(!trainParameter(epochSize
        , batchSize
        , errorTag
        , optimizationTag
        , trainingInput
        , trainingOutput
        , validationInput
        , validationOutput
        , concurrency
        , earlyStoppingLimit))
        return false;

    std::cout << "error: " << calculateError(testInput, testOutput, errorTag) << std::endl;

    return true;
}

template<class T>
bool NeuralNetwork<T>::predict(const Matrix<T> &input
    , Matrix<T> &output)
{
    if(!checkValidity(input))
        return false;

    if(!propagate(input, false))
        return false;
    
    output = mLayers.back()->data();
    return true;
}

template<class T>
bool NeuralNetwork<T>::checkValidity(std::size_t epochSize
    , std::size_t batchSize
    , ErrorTag errorTag
    , OptimizationTag optimizationTag
    , const std::vector<Matrix<T>> &trainingInput
    , const std::vector<Matrix<T>> &trainingOutput
    , const std::vector<Matrix<T>> &validationInput
    , const std::vector<Matrix<T>> &validationOutput
    , const std::vector<Matrix<T>> &testInput
    , const std::vector<Matrix<T>> &testOutput
    , std::size_t concurrency
    , std::size_t earlyStoppingLimit) const
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

    if(!checkMatrixValidity(trainingInput, mLayers.front()->data().column())
        || !checkMatrixValidity(trainingOutput, mLayers.back()->data().column())
        || !checkMatrixValidity(validationInput, mLayers.front()->data().column())
        || !checkMatrixValidity(validationOutput, mLayers.back()->data().column())
        || !checkMatrixValidity(testInput, mLayers.front()->data().column())
        || !checkMatrixValidity(testOutput, mLayers.back()->data().column()))
        return trainingError("data's elements does not match layer's io.");

    if(concurrency == 0ull)
        return trainingError("concurrency size is invalid.");

    if(earlyStoppingLimit == 0ull)
        return trainingError("condition of early stopping is invalid.");

    for(auto &&layer : mLayers)
        if(layer->column() == 0ull)
            return trainingError("layer has 0 size's input");

    return true;
}

template<class T>
bool NeuralNetwork<T>::checkValidity(const Matrix<T> &input) const
{
    if(mLayers.empty())
        return activationError("multi-layer perceptron has no layers");
    for(auto &&layer : mLayers)
        if(layer->column() == 0ull)
            return activationError("layer has 0 size's input.");

    if(input.row() != 1ull
        || input.column() != mLayers.front()->column())
        return activationError("input size does not match layer's input.");

    return true;
}

template<class T>
bool NeuralNetwork<T>::randomizeParameter()
{
    auto &&layerIter{std::next(mLayers.begin())};
    auto &&weightIter{mWeights.begin()};
    auto &&biasIter{mBiases.begin()};
    for(;
        layerIter != mLayers.end();
        layerIter++
            , weightIter++
            , biasIter++)
    {
        switch((*layerIter)->activationTag())
        {
            case(ActivationTag::NONE):
            case(ActivationTag::SIGMOID):
            case(ActivationTag::ELU):
            case(ActivationTag::SOFTMAX):
            case(ActivationTag::RELU):
            {
                std::normal_distribution<T> dist{static_cast<T>(0), std::sqrt(static_cast<T>(2) / (*std::prev(layerIter))->data().column())};
                for(std::size_t r{0ull}; r < (*weightIter)->data().row(); r++)
                    for(std::size_t c{0ull}; c < (*weightIter)->data().column(); c++)
                        (*weightIter)->data()(r, c) = dist(RANDOM.engine());

                (*biasIter)->data() = static_cast<T>(0.1);

                break;
            }
        }
    }
    
    return true;
}

template<class T>
bool NeuralNetwork<T>::trainParameter(std::size_t epochSize
    , std::size_t batchSize
    , ErrorTag errorTag
    , OptimizationTag optimizationTag
    , const std::vector<Matrix<T>> &trainingInput
    , const std::vector<Matrix<T>> &trainingOutput
    , const std::vector<Matrix<T>> &validationInput
    , const std::vector<Matrix<T>> &validationOutput
    , std::size_t concurrency
    , std::size_t earlyStoppingLimit)
{
    T minError{std::numeric_limits<T>::max()};
    std::size_t stoppingCount{0ull};
    std::list<std::shared_ptr<Weight<T>>> weightGradients;
    std::list<std::shared_ptr<Bias<T>>> biasGradients;
    std::list<std::shared_ptr<Weight<T>>> minWeights;
    std::list<std::shared_ptr<Bias<T>>> minBiases;
    std::list<Matrix<T>> weightAdamMs;
    std::list<Matrix<T>> biasAdamMs;
    std::list<Matrix<T>> weightAdamVs;
    std::list<Matrix<T>> biasAdamVs;
    for(auto &&weight : mWeights)
    {
        weightGradients.emplace_back(new Weight<T>{weight->data().row(), weight->data().column()});
        minWeights.emplace_back(new Weight<T>{weight->data().row(), weight->data().column()});
        weightAdamMs.emplace_back(weight->data().row(), weight->data().column());
        weightAdamVs.emplace_back(weight->data().row(), weight->data().column());
    }
    for(auto &&bias : mBiases)
    {
        biasGradients.emplace_back(new Bias<T>{bias->data().column()});
        minBiases.emplace_back(new Bias<T>{bias->data().column()});
        biasAdamMs.emplace_back(bias->data().row(), bias->data().column());
        biasAdamVs.emplace_back(bias->data().row(), bias->data().column());
    }

    for(std::size_t epoch{0ull}; epoch < epochSize; epoch++)
    {
        for(auto &&trainingIndices{createTrainingIndices(trainingInput.size(), batchSize)};
            !trainingIndices.empty();
            trainingIndices.pop_front())
        {
            std::cout << "remaining data of epoch " << epoch + 1ull << ": " << trainingIndices.size() << std::string(20, ' ') << '\r' << std::flush;

            std::size_t trainingIndex{trainingIndices.front()};
            if(!propagate(trainingInput[trainingIndex], true))
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
                , biasGradients
                , weightAdamMs
                , biasAdamMs
                , weightAdamVs
                , biasAdamVs))
                return false;
            
            for(auto &&layer : mLayers)
                layer->updateDropout();
            for(auto &&gradient : weightGradients)
                gradient->data() = static_cast<T>(0);
            for(auto &&gradient : biasGradients)
                gradient->data() = static_cast<T>(0);
        }

        auto &&error{calculateError(validationInput, validationOutput, errorTag)};
        if(epochSize < 10 || (epoch + 1ull) % (epochSize / 10ull) == 0)
            std::cout << "epoch " << epoch + 1ull << "'s error: " << error << std::endl;

        if(shouldStopEarly(error
            , minError
            , earlyStoppingLimit
            , stoppingCount
            , minWeights
            , minBiases))
        {
            std::cout << "early stopping has been activated."
                << "\n    reached epoch: " << epoch + 1ull << "/" << epochSize << std::endl;
            break;
        }
    }

    for(auto &&weight : mWeights)
        delete weight;
    for(auto &&bias : mBiases)
        delete bias;
    mWeights.clear();
    mBiases.clear();
    for(auto &&minWeight : minWeights)
        mWeights.push_back(new Weight{std::move(*minWeight)});
    for(auto &&minBias : minBiases)
        mBiases.push_back(new Bias{std::move(*minBias)});

    return true;
}

template<class T>
std::deque<std::size_t> NeuralNetwork<T>::createTrainingIndices(std::size_t trainingSize
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
bool NeuralNetwork<T>::propagate(const Matrix<T> &trainingInput
    , bool isTraining)
{
    auto &&layerIter{mLayers.begin()};
    auto &&weightIter{mWeights.begin()};
    auto &&biasIter{mBiases.begin()};

    if(!(*layerIter)->activate(trainingInput, isTraining))
        return false;

    for(layerIter++;
        layerIter != mLayers.end();
        layerIter++
            , weightIter++
            , biasIter++)
    {
        if(!(*layerIter)->activate((*std::prev(layerIter))->data()
            , (*weightIter)->data()
            , (*biasIter)->data()
            , isTraining))
            return false;
    }

    return true;
}

template<class T>
bool NeuralNetwork<T>::backpropagate(const Matrix<T> &trainingOutput
    , ErrorTag errorTag
    , std::list<std::shared_ptr<Weight<T>>> &weightGradients
    , std::list<std::shared_ptr<Bias<T>>> &biasGradients)
{
    // reverse iterators
    auto &&layerIter{mLayers.rbegin()};
    auto &&weightIter{mWeights.rbegin()};
    auto &&biasIter{mBiases.rbegin()};
    auto &&weightGradientIter{weightGradients.rbegin()};
    auto &&biasGradientIter{biasGradients.rbegin()};

    // output layer
    Matrix<T> error{(*layerIter)->error(FUNCTION::derivativeErrorFunction<T>(errorTag)(trainingOutput, (*layerIter)->data()))};
    (*weightGradientIter)->data() += matmul(~(*std::next(layerIter))->data(), error);
    (*biasGradientIter)->data() += error;

    // others
    for(layerIter++
            , weightGradientIter++
            , biasGradientIter++;
        std::next(layerIter) != mLayers.rend();
        layerIter++
            , weightIter++
            , biasIter++
            , weightGradientIter++
            , biasGradientIter++)
    {
        error = (*layerIter)->error(matmul(error, ~(*weightIter)->data()));
        (*weightGradientIter)->data() += matmul(~(*std::next(layerIter))->data(), error);
        (*biasGradientIter)->data() += error;
    }

    return true;
}

template<class T>
bool NeuralNetwork<T>::calculateAverage(std::size_t batchSize
    , std::list<std::shared_ptr<Weight<T>>> &weightGradients
    , std::list<std::shared_ptr<Bias<T>>> &biasGradients)
{
    T denominator{static_cast<T>(batchSize)};

    for(auto &&gradient : weightGradients)
        gradient->data() /= denominator;
    for(auto &&gradient : biasGradients)
        gradient->data() /= denominator;

    return true;
}

template<class T>
bool NeuralNetwork<T>::updateParameter(OptimizationTag optimizationTag
    , std::list<std::shared_ptr<Weight<T>>> &weightGradients
    , std::list<std::shared_ptr<Bias<T>>> &biasGradients
    , std::list<Matrix<T>> &weightAdamMs
    , std::list<Matrix<T>> &biasAdamMs
    , std::list<Matrix<T>> &weightAdamVs
    , std::list<Matrix<T>> &biasAdamVs)
{
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
                (*weightIter)->data()
                    = FUNCTION::optimizationNone((*weightIter)->data()
                        , (*weightGradientIter)->data());
                (*biasIter)->data()
                    = FUNCTION::optimizationNone((*biasIter)->data()
                        , (*biasGradientIter)->data());
                break;
            case(OptimizationTag::ADAM):
                (*weightIter)->data()
                    = FUNCTION::adam((*weightIter)->data()
                        , (*weightGradientIter)->data()
                        , *weightAdamMIter
                        , *weightAdamVIter);
                (*biasIter)->data()
                    = FUNCTION::adam((*biasIter)->data()
                        , (*biasGradientIter)->data()
                        , *biasAdamMIter
                        , *biasAdamVIter);
                break;
        }
    }

    return true;
}

template<class T>
T NeuralNetwork<T>::calculateError(const std::vector<Matrix<T>> &inputs
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
        propagate(*inputIter, false);
        error += errorFunction((*outputIter), mLayers.back()->data());
    }

    return error;
}

template<class T>
bool NeuralNetwork<T>::shouldStopEarly(T error
    , T &minError
    , std::size_t earlyStoppingLimit
    , std::size_t &stoppingCount
    , std::list<std::shared_ptr<Weight<T>>> &minWeights
    , std::list<std::shared_ptr<Bias<T>>> &minBiases)
{
    if(error < minError)
    {
        minError = error;
        stoppingCount = 0ull;
        minWeights.clear();
        minBiases.clear();
        for(auto &&weight : mWeights)
            minWeights.emplace_back(new Weight{*weight});
        for(auto &&bias : mBiases)
            minBiases.emplace_back(new Bias{*bias});
    }
    else
        stoppingCount++;

    return stoppingCount == earlyStoppingLimit;
}

template<class T>
bool NeuralNetwork<T>::trainingError(const std::string &what) const
{
    std::cerr << "NeuralNetwork::trainingError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}

template<class T>
bool NeuralNetwork<T>::activationError(const std::string &what) const
{
    std::cerr << "NeuralNetwork::activationError():"
        << "\n    what: " << what
        << std::endl;
    return false;
}

template<class T>
bool NeuralNetwork<T>::openingFileError(const std::filesystem::path &filepath) const
{
    std::cerr << "NeuralNetwork::openingFileError():\n"
        "    what: failed to open specific file.\n"
        "    file: " << filepath.string() << std::endl;
    return false;
}

template<class T>
bool NeuralNetwork<T>::writingError(const std::string &what
    , const std::filesystem::path &filepath) const
{
    std::cerr << "NeuralNetwork::writingError():\n"
        "    what: " << what
        << "\n    file: " << filepath.string() << std::endl;
    return false;
}

template<class T>
bool NeuralNetwork<T>::readingError(const std::string &what
    , const std::filesystem::path &filepath) const
{
    std::cerr << "NeuralNetwork::readingError():\n"
        "    what: " << what
        << "\n    file: " << filepath.string() << std::endl;
    return false;
}

}

#endif