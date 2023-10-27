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
    friend class Saver;
    friend class Loader;

    Layer(std::size_t column
        , ActivationTag tag
        , double dropoutRate = 0.0);

    std::size_t column() const noexcept
        {return mColumn;}
    ActivationTag activationTag() const noexcept
        {return mActivationTag;}
    double dropout() const noexcept
        {return mDropoutRate;}

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

    bool activate(const Matrix<T> &in
        , bool isTraining)
        {return isTraining ? activateForTraining(in) : activateForPrediction(in);}
    bool activate(const Matrix<T> &prevOut
        , const Matrix<T> &weight
        , const Matrix<T> &bias
        , bool isTraining)
        {return isTraining ? activateForTraining(prevOut, weight, bias) : activateForPrediction(prevOut, weight, bias);}

    Matrix<T> error(const Matrix<T> &dError) const;

private:
    bool activateForTraining(const Matrix<T> &in);
    bool activateForTraining(const Matrix<T> &prevOut
        , const Matrix<T> &weight
        , const Matrix<T> &bias)
        {return activateForTraining(matmul(prevOut, weight) + bias);}
    bool activateForPrediction(const Matrix<T> &in);
    bool activateForPrediction(const Matrix<T> &prevOut
        , const Matrix<T> &weight
        , const Matrix<T> &bias)
        {return activateForPrediction(matmul(prevOut, weight) * (static_cast<T>(1) - mDropoutRate) + bias);}
    
    std::size_t mColumn;
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
    : mColumn{column}
    , mActivationTag{tag}
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
    // mDropout = mDropout.apply([&](T t){return static_cast<T>(RANDOM.floating<T>() >= mDropoutRate);});
    for(std::size_t r{0ull}; r < mDropout.row(); r++)
        for(std::size_t c{0ull}; c < mDropout.column(); c++)
            mDropout(r, c) = static_cast<T>(RANDOM.floating<T>() >= static_cast<T>(mDropoutRate));
    return true;
}

template<class T>
bool Layer<T>::activateForTraining(const Matrix<T> &in)
{
    auto &&activationFunction{FUNCTION::activationFunction<T>(activationTag())};
    
    mInput = in;
    mOutput = activationFunction(mInput) * mDropout;
    return true;
}

template<class T>
bool Layer<T>::activateForPrediction(const Matrix<T> &in)
{
    auto &&activationFunction{FUNCTION::activationFunction<T>(activationTag())};
    
    mInput = in;
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
            result *= dError;
            break;
        case(ActivationTag::SOFTMAX):
            result = matmul(dError, result);
            break;
    }

    return result *= mDropout;
}

}

#endif