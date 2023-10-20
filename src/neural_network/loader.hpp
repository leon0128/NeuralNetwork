#ifndef NEURAL_NETWORK_LOADER_HPP
#define NEURAL_NETWORK_LOADER_HPP

#include <iostream>
#include <utility>

#include "matrix/matrix.hpp"
#include "neural_network.hpp"
#include "layer.hpp"
#include "parameter_base.hpp"
#include "weight.hpp"
#include "bias.hpp"

namespace NEURAL_NETWORK
{

class Loader
{
public:
    Loader() = delete;

    template<class T>
    static bool load(std::istream&
        , NeuralNetwork<T>&);

    template<class T>
    static bool load(std::istream&
        , Layer<T>&);

    template<class T>
    static bool load(std::istream&
        , ParameterBase<T>&);
    
    template<class T>
    static bool load(std::istream&
        , Weight<T>&);
    
    template<class T>
    static bool load(std::istream&
        , Bias<T>&);

    template<class T>
    static bool load(std::istream&
        , Matrix<T>&);

    template<class T>
    static T read(std::istream&);
};

template<class T>
bool Loader::load(std::istream &stream
    , NeuralNetwork<T> &nn)
{
    nn = NeuralNetwork<T>{};
    std::size_t size{read<std::size_t>(stream)};
    for(std::size_t i{0ull}; i < size; i++)
    {
        nn.mLayers.push_back(new Layer<T>{0ull, ActivationTag::NONE});
        if(!load(stream, *nn.mLayers.back()))
            return false;
    }

    size = read<std::size_t>(stream);
    for(std::size_t i{0ull}; i < size; i++)
    {
        nn.mWeights.push_back(new Weight<T>{0ull, 0ull});
        if(!load(stream, *nn.mWeights.back()))
            return false;
    }

    size = read<std::size_t>(stream);
    for(std::size_t i{0ull}; i < size; i++)
    {
        nn.mBiases.push_back(new Bias<T>{0ull});
        if(!load(stream, *nn.mBiases.back()))
            return false;
    }

    return static_cast<bool>(stream);
}

template<class T>
bool Loader::load(std::istream &stream
    , Layer<T> &layer)
{
    layer = Layer<T>{0ull, ActivationTag::NONE};
    layer.mColumn = read<std::size_t>(stream);
    layer.mActivationTag = read<ActivationTag>(stream);
    layer.mDropoutRate = read<double>(stream);

    return static_cast<bool>(stream);
}

template<class T>
bool Loader::load(std::istream &stream
    , ParameterBase<T> &pb)
{
    pb = ParameterBase<T>{0ull, 0ull};
    if(!load(stream, pb.mData))
        return false;

    return static_cast<bool>(stream);
}

template<class T>
bool Loader::load(std::istream &stream
    , Weight<T> &weight)
{
    ParameterBase<T> pb{0ull, 0ull};
    if(!load(stream, pb))
        return false;
    weight = Weight{std::move(pb)};

    return static_cast<bool>(stream);
}

template<class T>
bool Loader::load(std::istream &stream
    , Bias<T> &bias)
{
    ParameterBase<T> pb{0ull, 0ull};
    if(!load(stream, pb))
        return false;
    bias = Bias{std::move(pb)};

    return static_cast<bool>(stream);
}

template<class T>
bool Loader::load(std::istream &stream
    , Matrix<T> &matrix)
{
    std::size_t row{read<std::size_t>(stream)};
    std::size_t column{read<std::size_t>(stream)};
    matrix = Matrix<T>{row, column};
    for(std::size_t r{0ull}; r < row; r++)
        for(std::size_t c{0ull}; c < column; c++)
            matrix[r][c] = read<T>(stream);

    return static_cast<bool>(stream);
}

template<class T>
T Loader::read(std::istream &stream)
{
    T value;
    stream.seekg((alignof(T) - stream.tellg() % alignof(T)) % alignof(T)
        , std::ios_base::cur);
    stream.read(reinterpret_cast<char*>(&value)
        , sizeof(T));

    return value;
}

}

#endif