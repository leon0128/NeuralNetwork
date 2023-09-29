#ifndef NEURAL_NETWORK_LAYER_HPP
#define NEURAL_NETWORK_LAYER_HPP

#include "matrix/matrix.hpp"
#include "tag.hpp"

class Layer
{
public:
    Layer(std::size_t column
        , ActivationTag tag);

    auto &input()
        {return mInput;}
    const auto &input() const
        {return mInput;}
    auto &output()
        {return mOutput;}
    const auto &output() const
        {return mOutput;}

private:
    ActivationTag mActivationTag;
    Matrix<double> mInput;
    Matrix<double> mOutput;
};

#endif