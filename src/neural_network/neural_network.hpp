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
        , std::size_t earlyStopping = 5ull);

    bool predict(const Matrix<T> &input
        , Matrix<T> &output);

    bool save(const std::filesystem::path &filepath) const;
    bool load(const std::filesystem::path &filepath);

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
        , std::size_t earlyStopping) const;
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
        , std::size_t earlyStopping);
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
        , std::list<std::shared_ptr<Bias<T>>> &biasGradients);
    T calculateError(const std::vector<Matrix<T>> &inputs
        , const std::vector<Matrix<T>> &outputs
        , ErrorTag errorTag);

    template<class U>
    void writeValue(std::ofstream &stream
        , U &&value) const;
    template<class U>
    U readValue(std::ifstream &stream) const;

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
        Weight<T> *weight{new Weight<T>{lastLayer->output().column()
            , layer->input().column()}};
        Bias<T> *bias{new Bias<T>{layer->input().column()}};
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
    , std::size_t earlyStopping)
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
        , earlyStopping))
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
        , earlyStopping))
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
    
    output = mLayers.back()->output();
    return true;
}

template<class T>
bool NeuralNetwork<T>::save(const std::filesystem::path &filepath) const
{
    if(mLayers.empty())
        return writingError("NeuralNetwork has no layer.", filepath);

    std::ofstream stream{filepath, std::ios::out | std::ios::binary};
    if(!stream.is_open())
        return openingFileError(filepath);

    writeValue(stream, mLayers.size());
    for(auto &&layer : mLayers)
    {
        writeValue(stream, layer->input().column());
        writeValue(stream, layer->activationTag());
        writeValue(stream, layer->dropout());
    }

    for(auto &&weight : mWeights)
        for(std::size_t r{0ull}; r < weight->data().row(); r++)
            for(std::size_t c{0ull}; c < weight->data().column(); c++)
                writeValue(stream, weight->data()[r][c]);
    for(auto &&bias : mBiases)
        for(std::size_t c{0ull}; c < bias->data().column(); c++)
            writeValue(stream, bias->data()[0ull][c]);

    if(!stream)
        return writingError("failed to write parameters", filepath);
    
    return true;
}

template<class T>
bool NeuralNetwork<T>::load(const std::filesystem::path &filepath)
{
    if(!mLayers.empty())
        return readingError("NeuralNetwork has a layers already.", filepath);

    std::ifstream stream{filepath, std::ios::in | std::ios::binary};
    if(!stream.is_open())
        return openingFileError(filepath);

    for(std::size_t i{0ull}, numLayer{readValue<std::size_t>(stream)}; i < numLayer; i++)
    {
        std::size_t layerSize{readValue<std::size_t>(stream)};
        ActivationTag tag{readValue<ActivationTag>(stream)};
        double dropoutRate{readValue<double>(stream)};
        mLayers.push_back(new Layer<T>{layerSize, tag, dropoutRate});
    }

    if(mLayers.empty())
        return readingError("multi layer perceptron has no layer.", filepath);

    for(auto &&iter{mLayers.begin()}; std::next(iter) != mLayers.end(); iter++)
    {
        Weight<T> *weight{new Weight<T>{(*iter)->input().column()
            , (*std::next(iter))->input().column()}};
        for(std::size_t r{0ull}; r < weight->data().row(); r++)
            for(std::size_t c{0ull}; c < weight->data().column(); c++)
                weight->data()[r][c] = readValue<T>(stream);
        mWeights.push_back(weight);
    }

    for(auto &&iter{std::next(mLayers.begin())}; iter != mLayers.end(); iter++)
    {
        Bias<T> *bias{new Bias<T>{(*iter)->input().column()}};
        for(std::size_t c{0ull}; c < bias->data().column(); c++)
            bias->data()[0ull][c] = readValue<T>(stream);
        mBiases.push_back(bias);
    }

    if(!stream)
        return readingError("failed to read parameters.", filepath);

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
    , std::size_t earlyStopping) const
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

    if(earlyStopping == 0ull)
        return trainingError("condition of early stopping is invalid.");

    for(auto &&layer : mLayers)
        if(layer->input().column() == 0ull)
            return trainingError("layer has 0 size's input");

    return true;
}

template<class T>
bool NeuralNetwork<T>::checkValidity(const Matrix<T> &input) const
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
                std::normal_distribution<T> dist{0.0, std::sqrt(2.0 / (*std::prev(layerIter))->output().column())};
                for(std::size_t r{0ull}; r < (*weightIter)->data().row(); r++)
                    for(std::size_t c{0ull}; c < (*weightIter)->data().column(); c++)
                        (*weightIter)->data()[r][c] = dist(RANDOM.engine());
                for(std::size_t c{0ull}; c < (*biasIter)->data().column(); c++)
                    (*biasIter)->data()[0ull][c] = 0.1;

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
    , std::size_t earlyStopping)
{
    T minError{std::numeric_limits<T>::max()};
    std::size_t stoppingCount{0ull};
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
                , biasGradients))
                return false;
            
            for(auto &&layer : mLayers)
                layer->updateDropout();
            for(auto &&gradient : weightGradients)
                gradient->data().apply([](T in){return T{0};});
            for(auto &&gradient : biasGradients)
                gradient->data().apply([](T in){return T{0};});
        }

        auto &&error{calculateError(validationInput, validationOutput, errorTag)};
        if(epochSize < 10 || (epoch + 1ull) % (epochSize / 10ull) == 0)
            std::cout << "epoch " << epoch + 1ull << "'s error: " << error << std::endl;

        stoppingCount
            = error < minError
                ? (minError = error, 0ull) // assign and return 0
                : stoppingCount + 1ull;

        if(stoppingCount == earlyStopping)
        {
            std::cout << "early stopping has been activated."
                << "\n    reached epoch: " << epoch + 1ull << "/" << epochSize << std::endl;
            break;
        }
    }

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
        if(!(*layerIter)->activate((*std::prev(layerIter))->output()
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
    Matrix<T> error{(*layerIter)->error(FUNCTION::derivativeErrorFunction<T>(errorTag)(trainingOutput, (*layerIter)->output()))};
    (*weightGradientIter)->data() += ~(*std::next(layerIter))->output() * error;
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
        error = (*layerIter)->error(error * ~(*weightIter)->data());
        (*weightGradientIter)->data() += ~(*std::next(layerIter))->output() * error;
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
        gradient->data().apply([&](T in){return in / denominator;});
    for(auto &&gradient : biasGradients)
        gradient->data().apply([&](T in){return in / denominator;});

    return true;
}

template<class T>
bool NeuralNetwork<T>::updateParameter(OptimizationTag optimizationTag
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
        propagate(*inputIter, true);
        error += errorFunction((*outputIter), mLayers.back()->output());
    }

    return error;
}

template<class T>
template<class U>
void NeuralNetwork<T>::writeValue(std::ofstream &stream
    , U &&value) const
{
    static const char padding[256]{0};

    stream.write(reinterpret_cast<const char*>(&padding)
        , (alignof(U) - stream.tellp() % alignof(U)) % alignof(U));
    stream.write(reinterpret_cast<const char*>(&value)
        , sizeof(U));
}

template<class T>
template<class U>
U NeuralNetwork<T>::readValue(std::ifstream &stream) const
{
    U value;
    stream.seekg((alignof(U) - stream.tellg() % alignof(U)) % alignof(U)
        , std::ios_base::cur);
    stream.read(reinterpret_cast<char*>(&value)
        , sizeof(U));
    return value;
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