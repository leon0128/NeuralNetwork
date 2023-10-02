#ifndef NEURAL_NETWORK_LAYER_HPP
#define NEURAL_NETWORK_LAYER_HPP

#include <cmath>

#include "matrix/matrix.hpp"
#include "tag.hpp"

class Layer
{
public:
    Layer(std::size_t column
        , ActivationTag tag);

    ActivationTag activationTag() const noexcept
        {return mActivationTag;}

    bool activate();

    auto &input()
        {return mInput;}
    const auto &input() const
        {return mInput;}
    auto &output()
        {return mOutput;}
    const auto &output() const
        {return mOutput;}

private:
    double activateNone(double in) const
        {return in;}
    double activateElu(double in) const
        {return in >= 0.0 ? in : std::exp(in) - 1.0;}

    ActivationTag mActivationTag;
    Matrix<double> mInput;
    Matrix<double> mOutput;
};

#endif