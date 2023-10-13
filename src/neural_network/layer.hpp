#ifndef NEURAL_NETWORK_LAYER_HPP
#define NEURAL_NETWORK_LAYER_HPP

#include "matrix/matrix.hpp"
#include "function.hpp"
#include "tag.hpp"

template<class T>
class Layer
{
public:
    Layer(std::size_t column
        , ActivationTag tag);

    ActivationTag activationTag() const noexcept
        {return mActivationTag;}

    bool activate();

    void input(const Matrix<T> &other)
        {mInput = other;}
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
    Matrix<T> mInput;
    Matrix<T> mOutput;
};

template<class T>
Layer<T>::Layer(std::size_t column
    , ActivationTag tag)
    : mActivationTag{tag}
    , mInput{1ull, column}
    , mOutput{1ull, column}
{
}

template<class T>
bool Layer<T>::activate()
{
    auto &&activationFunction{FUNCTION::activationFunction<T>(activationTag())};
    output() = activationFunction(input());
    return true;
}

#endif