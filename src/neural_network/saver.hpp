#ifndef NEURAL_NETWORK_SAVER_HPP
#define NEURAL_NETWORK_SAVER_HPP

#include <ostream>

#include "matrix/matrix.hpp"
#include "neural_network.hpp"
#include "layer.hpp"
#include "parameter_base.hpp"
#include "bias.hpp"
#include "weight.hpp"

namespace NEURAL_NETWORK
{

template<class T>
class NeuralNetwork;
template<class T>
class Layer;
template<class T>
class ParameterBase;
template<class T>
class Weight;
template<class T>
class Bias;

class Saver
{
public:
    Saver() = delete;

    template<class T>
    static bool save(std::ostream&
        , const NeuralNetwork<T>&);

    template<class T>
    static bool save(std::ostream&
        , const Layer<T>&);

    template<class T>
    static bool save(std::ostream&
        , const ParameterBase<T>&);

    template<class T>
    static bool save(std::ostream&
        , const Matrix<T>&);

    template<class T>
    static void write(std::ostream &stream
        , T &&value);
};

template<class T>
bool Saver::save(std::ostream &stream
    , const NeuralNetwork<T> &nn)
{
    write(stream, nn.mLayers.size());
    for(auto &&layer : nn.mLayers)
        if(!save(stream, *layer))
            return false;
    write(stream, nn.mWeights.size());
    for(auto &&weight : nn.mWeights)
        if(!save(stream, *weight))
            return false;
    write(stream, nn.mBiases.size());
    for(auto &&bias : nn.mBiases)
        if(!save(stream, *bias))
            return false;

    return static_cast<bool>(stream);
}

template<class T>
bool Saver::save(std::ostream &stream
    , const Layer<T> &layer)
{
    write(stream, layer.mColumn);
    write(stream, layer.mActivationTag);
    write(stream, layer.mDropoutRate);

    return static_cast<bool>(stream);
}

template<class T>
bool Saver::save(std::ostream &stream
    , const ParameterBase<T> &pb)
{
    if(!save(stream, pb.mData))
        return false;

    return static_cast<bool>(stream);
}

template<class T>
bool Saver::save(std::ostream &stream
    , const Matrix<T> &matrix)
{
    write(stream, matrix.row());
    write(stream, matrix.column());
    for(std::size_t r{0ull}; r < matrix.row(); r++)
        for(std::size_t c{0ull}; c < matrix.column(); c++)
            write(stream, matrix(r, c));

    return static_cast<bool>(stream);
}

template<class T>
void Saver::write(std::ostream &stream
    , T &&value)
{
    static const char padding[256]{0};
    stream.write(reinterpret_cast<const char*>(&padding)
        , (alignof(T) - stream.tellp() % alignof(T)) % alignof(T));
    stream.write(reinterpret_cast<const char*>(&value)
        , sizeof(T));
}

}

#endif