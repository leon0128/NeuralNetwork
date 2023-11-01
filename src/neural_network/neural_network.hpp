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
#include <mutex>

#include "saver.hpp"
#include "loader.hpp"
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

class Saver;
class Loader;
template<class T>
class Layer;
template<class T>
class ParameterBase;
template<class T>
class Weight;
template<class T>
class Bias;

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
        , std::size_t earlyStoppingLimit = 5ull
        , std::size_t concurrency = 8ull);

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
        , std::size_t earlyStoppingLimit
        , std::size_t concurrency) const;
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
        , const std::vector<Matrix<T>> &testInput
        , const std::vector<Matrix<T>> &testOutput
        , std::size_t earlyStoppingLimit
        , std::size_t concurrency);
    std::deque<std::size_t> createTrainingIndices(std::size_t trainingSize
        , std::size_t batchSize) const;
    bool propagate(std::list<Layer<T>*> &layers
        , const Matrix<T> &trainingInput
        , bool isTraining);
    bool backpropagate(const std::list<Layer<T>*> &layers
        , const Matrix<T> &trainingOutput
        , ErrorTag errorTag
        , std::list<Weight<T>*> &weightGradients
        , std::list<Bias<T>*> &biasGradients
        , std::mutex &weightGradientsMutex
        , std::mutex &biasGradientsMutex);
    bool calculateAverage(std::size_t batchSize
        , std::list<Weight<T>*> &weightGradients
        , std::list<Bias<T>*> &biasGradients);
    bool updateParameter(OptimizationTag optimizationTag
        , std::list<Weight<T>*> &weightGradients
        , std::list<Bias<T>*> &biasGradients
        , const std::filesystem::path &adamFilepath);
    T calculateError(std::vector<std::list<Layer<T>*>> &concurrencyLayers
        , const std::vector<Matrix<T>> &inputs
        , const std::vector<Matrix<T>> &outputs
        , ErrorTag errorTag
        , std::size_t concurrency);
    bool shouldStopEarly(T error
        , T &minError
        , std::size_t earlyStoppingLimit
        , std::size_t &stoppingCount
        , const std::filesystem::path &minParameterFilepath);
    
    bool saveParameter(const std::filesystem::path &filepath) const;
    bool loadParameter(const std::filesystem::path &filepath);
    bool saveAdam(const std::filesystem::path &filepath
        , const std::list<Matrix<T>> &weightMs
        , const std::list<Matrix<T>> &weightVs
        , const std::list<Matrix<T>> &biasMs
        , const std::list<Matrix<T>> &biasVs) const;
    bool loadAdam(const std::filesystem::path &filepath
        , std::list<Matrix<T>> &weightMs
        , std::list<Matrix<T>> &weightVs
        , std::list<Matrix<T>> &biasMs
        , std::list<Matrix<T>> &biasVs) const;

    bool trainingError(const std::string &what) const;
    bool activationError(const std::string &what) const;
    bool openingFileError(const std::filesystem::path &filepath) const;
    bool savingError(const std::string &what
        , const std::filesystem::path &filepath) const;
    bool loadingError(const std::string &what
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
    , std::size_t earlyStoppingLimit
    , std::size_t concurrency)
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
        , earlyStoppingLimit
        , concurrency))
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
        , testInput
        , testOutput
        , earlyStoppingLimit
        , concurrency))
        return false;

    return true;
}

template<class T>
bool NeuralNetwork<T>::predict(const Matrix<T> &input
    , Matrix<T> &output)
{
    if(!checkValidity(input))
        return false;

    if(!propagate(mLayers, input, false))
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
    , std::size_t earlyStoppingLimit
    , std::size_t concurrency) const
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

    if(earlyStoppingLimit == 0ull)
        return trainingError("condition of early stopping is invalid.");

    if(concurrency == 0ull)
        return trainingError("concurrency size is invalid.");

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
    , const std::vector<Matrix<T>> &testInput
    , const std::vector<Matrix<T>> &testOutput
    , std::size_t earlyStoppingLimit
    , std::size_t concurrency)
{
    T minError{std::numeric_limits<T>::max()};
    std::size_t stoppingCount{0ull};
    std::filesystem::path minParameterFilepath{std::filesystem::temp_directory_path() / std::filesystem::path{"neural_network_"}.concat(std::to_string(RANDOM()))};
    std::filesystem::path adamFilepath{std::filesystem::temp_directory_path() / std::filesystem::path{"neural_network_"}.concat(std::to_string(RANDOM()))};
    std::list<Weight<T>*> weightGradients;
    std::list<Bias<T>*> biasGradients;
    std::vector<std::list<Layer<T>*>> concurrencyLayers(concurrency, std::list<Layer<T>*>(mLayers.size()));
    std::deque<std::size_t> allTrainingIndices;
    std::deque<std::size_t> batchTrainingIndices;
    std::mutex weightGradientsMutex;
    std::mutex biasGradientsMutex;
    std::mutex trainingIndicesMutex;

    for(auto &&weight : mWeights)
        weightGradients.emplace_back(new Weight<T>{weight->data().row(), weight->data().column()});
    for(auto &&bias : mBiases)
        biasGradients.emplace_back(new Bias<T>{bias->data().column()});
    for(auto &&layers : concurrencyLayers)
        for(auto &&layer : layers)
            layer = new Layer<T>{0ull, ActivationTag::NONE};

    auto &&destruct{[&]()->void
        {
            for(auto &&weight : weightGradients)
                delete weight;
            for(auto &&bias : biasGradients)
                delete bias;
            for(auto &&layers : concurrencyLayers)
                for(auto &&layer : layers)
                    delete layer;
        }};
    auto &&copy{[&]()->void
        {
            for(auto &&conLayers : concurrencyLayers)
                for(auto &&iter{mLayers.begin()}, conIter{conLayers.begin()};
                    iter != mLayers.end();
                    iter++,conIter++)
                    **conIter = **iter;
        }};
    auto &&transfer{[&]()->void
        {
            batchTrainingIndices.resize(batchSize);
            std::copy(allTrainingIndices.begin()
                , allTrainingIndices.begin() + batchSize
                , batchTrainingIndices.begin());
            allTrainingIndices.erase(allTrainingIndices.begin()
                , allTrainingIndices.begin() + batchSize);
        }};
    auto &&propagateAndBackpropagate{[&](std::size_t idx)->bool
        {
            std::unique_lock lock{trainingIndicesMutex};
            if(batchTrainingIndices.empty())
                return false;
            std::size_t trainingIndex{batchTrainingIndices.front()};
            batchTrainingIndices.pop_front();
            lock.unlock();

            if(!propagate(concurrencyLayers[idx], trainingInput[trainingIndex], true))
                std::runtime_error("propagate error");
            if(!backpropagate(concurrencyLayers[idx]
                , trainingOutput[trainingIndex]
                , errorTag
                , weightGradients
                , biasGradients
                , weightGradientsMutex
                , biasGradientsMutex))
                std::runtime_error("backpropagate error");

            return true;
        }};

    for(std::size_t epoch{0ull}; epoch < epochSize; epoch++)
    {
        for(allTrainingIndices = createTrainingIndices(trainingInput.size(), batchSize);
            !allTrainingIndices.empty();)
        {
            std::cout << "remaining data of epoch " << epoch + 1ull << ": " << allTrainingIndices.size() << std::string(10, ' ') << '\r' << std::flush;

            copy();
            transfer();
            CONCURRENCY::execute(propagateAndBackpropagate, concurrency);
            
            if(!calculateAverage(batchSize
                , weightGradients
                , biasGradients))
                return destruct(), false;
            if(!updateParameter(optimizationTag
                , weightGradients
                , biasGradients
                , adamFilepath))
                return destruct(), false;
            
            for(auto &&layer : mLayers)
                layer->updateDropout();
            for(auto &&gradient : weightGradients)
                gradient->data() = static_cast<T>(0);
            for(auto &&gradient : biasGradients)
                gradient->data() = static_cast<T>(0);
        }

        auto &&error{calculateError(concurrencyLayers, validationInput, validationOutput, errorTag, concurrency)};
        if(epochSize < 10 || (epoch + 1ull) % (epochSize / 10ull) == 0)
            std::cout << "epoch " << epoch + 1ull << "'s error: " << error << std::string(10, ' ') << std::endl;

        if(shouldStopEarly(error
            , minError
            , earlyStoppingLimit
            , stoppingCount
            , minParameterFilepath))
        {
            std::cout << "early stopping has been activated."
                << "\n    reached epoch: " << epoch + 1ull << "/" << epochSize << std::endl;
            break;
        }
    }

    if(!loadParameter(minParameterFilepath))
        return destruct(), false;

    std::cout << "error: " << calculateError(concurrencyLayers, testInput, testOutput, errorTag, concurrency) << std::endl;

    return destruct(), true;
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
bool NeuralNetwork<T>::propagate(std::list<Layer<T>*> &layers
    , const Matrix<T> &trainingInput
    , bool isTraining)
{
    auto &&layerIter{layers.begin()};
    auto &&weightIter{mWeights.begin()};
    auto &&biasIter{mBiases.begin()};

    if(!(*layerIter)->activate(trainingInput, isTraining))
        return false;

    for(layerIter++;
        layerIter != layers.end();
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
bool NeuralNetwork<T>::backpropagate(const std::list<Layer<T>*> &layers
    , const Matrix<T> &trainingOutput
    , ErrorTag errorTag
    , std::list<Weight<T>*> &weightGradients
    , std::list<Bias<T>*> &biasGradients
    , std::mutex &weightGradientsMutex
    , std::mutex &biasGradientsMutex)
{
    // reverse iterators
    auto &&layerIter{layers.rbegin()};
    auto &&weightIter{mWeights.rbegin()};
    auto &&biasIter{mBiases.rbegin()};
    auto &&weightGradientIter{weightGradients.rbegin()};
    auto &&biasGradientIter{biasGradients.rbegin()};

    std::unique_lock weightLock{weightGradientsMutex, std::defer_lock};
    std::unique_lock biasLock{biasGradientsMutex, std::defer_lock};

    // output layer
    Matrix<T> error{(*layerIter)->error(FUNCTION::derivativeErrorFunction<T>(errorTag)(trainingOutput, (*layerIter)->data()))};
    Matrix<T> weightError{matmul(~(*std::next(layerIter))->data(), error)};
    weightLock.lock();
    (*weightGradientIter)->data() += weightError;
    weightLock.unlock();
    biasLock.lock();
    (*biasGradientIter)->data() += error;
    biasLock.unlock();

    // others
    for(layerIter++
            , weightGradientIter++
            , biasGradientIter++;
        std::next(layerIter) != layers.rend();
        layerIter++
            , weightIter++
            , biasIter++
            , weightGradientIter++
            , biasGradientIter++)
    {
        error = (*layerIter)->error(matmul(error, ~(*weightIter)->data()));
        weightError = matmul(~(*std::next(layerIter))->data(), error);
        weightLock.lock();
        (*weightGradientIter)->data() += weightError;
        weightLock.unlock();
        biasLock.lock();
        (*biasGradientIter)->data() += error;
        biasLock.unlock();
    }

    return true;
}

template<class T>
bool NeuralNetwork<T>::calculateAverage(std::size_t batchSize
    , std::list<Weight<T>*> &weightGradients
    , std::list<Bias<T>*> &biasGradients)
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
    , std::list<Weight<T>*> &weightGradients
    , std::list<Bias<T>*> &biasGradients
    , const std::filesystem::path &adamFilepath)
{
    auto &&weightIter{mWeights.begin()};
    auto &&biasIter{mBiases.begin()};
    auto &&weightGradientIter{weightGradients.begin()};
    auto &&biasGradientIter{biasGradients.begin()};

    switch(optimizationTag)
    {
        case(OptimizationTag::NONE):
        {
            for(;
                weightIter != mWeights.end();
                weightIter++
                    , biasIter++
                    , weightGradientIter++
                    , biasGradientIter++)
            {
                (*weightIter)->data()
                    = FUNCTION::optimizationNone((*weightIter)->data()
                        , (*weightGradientIter)->data());
                (*biasIter)->data()
                    = FUNCTION::optimizationNone((*biasIter)->data()
                        , (*biasGradientIter)->data());
            }
            break;
        }
        case(OptimizationTag::ADAM):
        {
            std::list<Matrix<T>> weightAdamMs(mWeights.size());
            std::list<Matrix<T>> weightAdamVs(mWeights.size());
            std::list<Matrix<T>> biasAdamMs(mBiases.size());
            std::list<Matrix<T>> biasAdamVs(mBiases.size());

            if(!loadAdam(adamFilepath
                , weightAdamMs
                , weightAdamVs
                , biasAdamMs
                , biasAdamVs))
                return false;

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
            }

            if(!saveAdam(adamFilepath
                , weightAdamMs
                , weightAdamVs
                , biasAdamMs
                , biasAdamVs))
                return false;
        }
    }

    return true;
}

template<class T>
T NeuralNetwork<T>::calculateError(std::vector<std::list<Layer<T>*>> &concurrencyLayers
    , const std::vector<Matrix<T>> &inputs
    , const std::vector<Matrix<T>> &outputs
    , ErrorTag errorTag
    , std::size_t concurrency)
{
    auto &&errorFunction{FUNCTION::errorFunction<T>(errorTag)};

    Matrix<T> error{1ull, inputs.size()};
    std::size_t ioIdx{};
    std::mutex ioIdxMutex;

    auto &&calculate{[&](std::size_t threadIdx)->bool
        {
            std::unique_lock ioIdxLock{ioIdxMutex};
            std::size_t idx{ioIdx++};
            ioIdxLock.unlock();

            if(idx >= inputs.size())
                return false;
            
            if(!propagate(concurrencyLayers[threadIdx], inputs[idx], false))
                std::runtime_error("propagate error");
            error(0, idx) = errorFunction(outputs[idx], concurrencyLayers[threadIdx].back()->data());
            return true;
        }};

    CONCURRENCY::execute(calculate, concurrency);

    return error.sum();
}

template<class T>
bool NeuralNetwork<T>::shouldStopEarly(T error
    , T &minError
    , std::size_t earlyStoppingLimit
    , std::size_t &stoppingCount
    , const std::filesystem::path &minParameterFilepath)
{
    if(error < minError)
    {
        minError = error;
        stoppingCount = 0ull;
        
        if(!saveParameter(minParameterFilepath))
            return false;
    }
    else
        stoppingCount++;

    return stoppingCount == earlyStoppingLimit;
}

template<class T>
bool NeuralNetwork<T>::saveParameter(const std::filesystem::path &filepath) const
{
    std::ofstream stream{filepath, std::ios_base::out | std::ios_base::binary};
    if(!stream.is_open())
        return openingFileError(filepath);
    for(auto &&weight : mWeights)
        if(!Saver::save(stream, *weight))
            return savingError("failed to save weight to file.", filepath);
    for(auto &&bias : mBiases)
        if(!Saver::save(stream, *bias))
            return savingError("failed to save bias to file.", filepath);

    return true;
}

template<class T>
bool NeuralNetwork<T>::loadParameter(const std::filesystem::path &filepath)
{
    std::ifstream stream{filepath, std::ios_base::in | std::ios_base::binary};
    if(!stream.is_open())
        return openingFileError(filepath);
    for(auto &&weight : mWeights)
        if(!Loader::load(stream, *weight))
            return loadingError("failed to load weight from file.", filepath);
    for(auto &&bias : mBiases)
        if(!Loader::load(stream, *bias))
            return loadingError("failed to load bias from file.", filepath);

    return true;
}

template<class T>
bool NeuralNetwork<T>::saveAdam(const std::filesystem::path &filepath
    , const std::list<Matrix<T>> &weightMs
    , const std::list<Matrix<T>> &weightVs
    , const std::list<Matrix<T>> &biasMs
    , const std::list<Matrix<T>> &biasVs) const
{
    std::ofstream stream{filepath, std::ios_base::out | std::ios_base::binary};
    if(!stream.is_open())
        return openingFileError(filepath);
    
    for(auto &&m : weightMs)
        if(!Saver::save(stream, m))
            return savingError("failed to save adam parameter to file.", filepath);
    for(auto &&v : weightVs)
        if(!Saver::save(stream, v))
            return savingError("failed to save adam parameter to file.", filepath);
    for(auto &&m : biasMs)
        if(!Saver::save(stream, m))
            return savingError("failed to save adam parameter to file.", filepath);
    for(auto &&v : biasVs)
        if(!Saver::save(stream, v))
            return savingError("failed to save adam parameter to file.", filepath);

    return true;
}

template<class T>
bool NeuralNetwork<T>::loadAdam(const std::filesystem::path &filepath
    , std::list<Matrix<T>> &weightMs
    , std::list<Matrix<T>> &weightVs
    , std::list<Matrix<T>> &biasMs
    , std::list<Matrix<T>> &biasVs) const
{
    std::ifstream stream{filepath, std::ios_base::in | std::ios_base::binary};
    if(!stream.is_open())
    {
        auto &&weightIter{mWeights.begin()};
        auto &&biasIter{mBiases.begin()};
        auto &&weightMsIter{weightMs.begin()};
        auto &&weightVsIter{weightVs.begin()};
        auto &&biasMsIter{biasMs.begin()};
        auto &&biasVsIter{biasVs.begin()};
        for(;
            weightIter != mWeights.end();
            weightIter++
                , biasIter++
                , weightMsIter++
                , weightVsIter++
                , biasMsIter++
                , biasVsIter++)
        {
            *weightMsIter = Matrix<T>{(*weightIter)->data().row(), (*weightIter)->data().column()};
            *weightVsIter = Matrix<T>{(*weightIter)->data().row(), (*weightIter)->data().column()};
            *biasMsIter = Matrix<T>{(*biasIter)->data().row(), (*biasIter)->data().column()};
            *biasVsIter = Matrix<T>{(*biasIter)->data().row(), (*biasIter)->data().column()};
        }
    }
    else
    {
        for(auto &&m : weightMs)
            if(!Loader::load(stream, m))
                return loadingError("failed to load adam parameter from file.", filepath);
        for(auto &&v : weightVs)
            if(!Loader::load(stream, v))
                return loadingError("failed to load adam parameter from file.", filepath);
        for(auto &&m : biasMs)
            if(!Loader::load(stream, m))
                return loadingError("failed to load adam parameter from file.", filepath);
        for(auto &&v : biasVs)
            if(!Loader::load(stream, v))
                return loadingError("failed to load adam parameter from file.", filepath);
    }

    return true;
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
bool NeuralNetwork<T>::savingError(const std::string &what
    , const std::filesystem::path &filepath) const
{
    std::cerr << "NeuralNetwork::savingError():\n"
        "    what: " << what
        << "\n    file: " << filepath.string() << std::endl;
    return false;
}

template<class T>
bool NeuralNetwork<T>::loadingError(const std::string &what
    , const std::filesystem::path &filepath) const
{
    std::cerr << "NeuralNetwork::loadingError():\n"
        "    what: " << what
        << "\n    file: " << filepath.string() << std::endl;
    return false;
}

}

#endif