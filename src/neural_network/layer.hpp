#ifndef NEURAL_NETWORK_LAYER_HPP
#define NEURAL_NETWORK_LAYER_HPP

#include "matrix/matrix.hpp"
#include "random.hpp"
#include "function.hpp"
#include "tag.hpp"

namespace NEURAL_NETWORK
{

template<class T>
class Layer
{
public:
    Layer(std::size_t column
        , ActivationTag tag
        , double dropoutRate = 1.0);

    ActivationTag activationTag() const noexcept
        {return mActivationTag;}

    auto &input()
        {return mInput;}
    const auto &input() const
        {return mInput;}
    auto &output()
        {return mOutput;}
    const auto &output() const
        {return mOutput;}

    // this function is called by constructor
    bool updateDropout();
    auto &&dropout() const noexcept
        {return mDropout;}

    bool activate(const Matrix<T> &input);
    bool activate(const Matrix<T> &prevOutput
        , const Matrix<T> &weight
        , const Matrix<T> &bias)
        {return activate(prevOutput * weight + bias);}
    
    Matrix<T> error(const Matrix<T> &dError) const;

private:
    ActivationTag mActivationTag;
    Matrix<T> mInput;
    Matrix<T> mOutput;
    double mDropoutRate;
    Matrix<T> mDropout;
};

template<class T>
Layer<T>::Layer(std::size_t column
    , ActivationTag tag
    , double dropoutRate)
    : mActivationTag{tag}
    , mInput{1ull, column}
    , mOutput{1ull, column}
    , mDropoutRate{dropoutRate}
    , mDropout{1ull, column}
{
    updateDropout();
}

template<class T>
bool Layer<T>::updateDropout()
{
    mDropout.apply([&](T in){return static_cast<T>(RANDOM.floating() <= mDropoutRate);});
    return true;
}

template<class T>
bool Layer<T>::activate(const Matrix<T> &input)
{
    auto &&activationFunction{FUNCTION::activationFunction<T>(activationTag())};
    
    mInput = input;
    mOutput = activationFunction(mInput);
    return true;
}

template<class T>
Matrix<T> Layer<T>::error(const Matrix<T> &dError) const
{
    Matrix<T> result{FUNCTION::derivativeActivationFunction<T>(activationTag())(output())};

    switch(activationTag())
    {
        case(ActivationTag::NONE):
        case(ActivationTag::ELU):
        case(ActivationTag::SIGMOID):
        case(ActivationTag::RELU):
            for(std::size_t c{0ull}; c < result.column(); c++)
                result[0ull][c] *= dError[0ull][c];
            break;
        case(ActivationTag::SOFTMAX):
            result = dError * result;
            break;
    }

    return result;
}

}

#endif