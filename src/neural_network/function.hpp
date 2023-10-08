#ifndef NEURAL_NETWORK_FUNCTION_HPP
#define NEURAL_NETWORK_FUNCTION_HPP

#include <cmath>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <limits>

#include "tag.hpp"

namespace FUNCTION
{

inline const double eluAlpha{1.0};

inline double softmaxSum{0.0};

inline const double optimizationNoneLearningRate{0.01};

inline const double adamLearningRate{0.001};
inline const double adamBeta1{0.9};
inline const double adamBeta2{0.999};
inline const double adamEpsilon{1.0e-7};

template<class T>
inline T activationNone(T input)
    {return input;}

template<class T>
inline T elu(T input)
    {return input >= T{0} ? input : std::exp(input) - T{eluAlpha};}

template<class T>
inline T sigmoid(T input)
    {return T{1} / (T{1} + std::exp(T{-1} * input));}

template<class T>
inline T relu(T input)
    {return input >= T{0} ? input : T{0};}

template<class T>
inline T softmax(T input)
    {return std::exp(input) / softmaxSum;}

template<class T>
inline T mse(T teacher, T output)
    {return std::pow(teacher - output, T{2}) / T{2};}

template<class T>
inline T crossEntropy(T teacher, T output)
    {return T{-1} * teacher * std::log(output);}

template<class T>
inline T differentiatedActivationNone(T output)
    {return T{1};}

template<class T>
inline T differentiatedElu(T output)
    {return output >= T{0} ? T{1} : output + eluAlpha;}

template<class T>
inline T differentiatedSigmoid(T output)
    {return output * (T{1} - output);}

template<class T>
inline T differentiatedRelu(T output)
    {return output >= T{0} ? T{1} : T{0};}

template<class T>
inline T differentiatedSoftmax(T output)
    {return output * (T{1} - output);}

template<class T>
inline T differentiatedMse(T teacher, T output)
    {return output - teacher;}

template<class T>
inline T differentiatedCrossEntropy(T teacher, T output)
    {return T{-1} * teacher / output;}

template<class T>
inline T optimizationNone(T prev, T gradient)
    {return prev - gradient * optimizationNoneLearningRate;}

template<class T>
inline T adam(T prev, T gradient)
{
    static T m{T{0}};
    static T v{T{0}};

    m = adamBeta1 * m + (T{1} - adamBeta1) * gradient;
    v = adamBeta2 * v + (T{1} - adamBeta2) * std::pow(gradient, T{2});
    double mHat{m / (T{1} - adamBeta1)};
    double vHat{v / (T{1} - adamBeta2)};
    return prev - (adamLearningRate * mHat / (std::sqrt(vHat) + adamEpsilon));
}

template<class T>
inline std::function<T(T)> activationFunction(ActivationTag tag)
{
    switch(tag)
    {
        case(ActivationTag::NONE):
            return activationNone<T>;
        case(ActivationTag::ELU):
            return elu<T>;
        case(ActivationTag::SIGMOID):
            return sigmoid<T>;
        case(ActivationTag::RELU):
            return relu<T>;
        case(ActivationTag::SOFTMAX):
            return softmax<T>;
    }

    return {};
}

template<class T>
inline std::function<T(T, T)> errorFunction(ErrorTag tag)
{
    switch(tag)
    {
        case(ErrorTag::MSE):
            return mse<T>;
        case(ErrorTag::CROSS_ENTROPY):
            return crossEntropy<T>;
        case(ErrorTag::NONE):
            throw std::runtime_error{"invalid error tag"};
    }

    return {};
}

template<class T>
inline std::function<T(T, T)> optimizationFunction(OptimizationTag tag)
{
    switch(tag)
    {
        case(OptimizationTag::ADAM):
            return adam<T>;
        case(OptimizationTag::NONE):
            return optimizationNone<T>;
    }

    return {};
}

template<class T>
inline std::function<T(T)> differentiatedActivationFunction(ActivationTag tag)
{
    switch(tag)
    {
        case(ActivationTag::NONE):
            return differentiatedActivationNone<T>;
        case(ActivationTag::ELU):
            return differentiatedElu<T>;
        case(ActivationTag::SIGMOID):
            return differentiatedSigmoid<T>;
        case(ActivationTag::RELU):
            return differentiatedRelu<T>;
        case(ActivationTag::SOFTMAX):
            return differentiatedSoftmax<T>;
    }

    return {};
}

template<class T>
inline std::function<T(T, T)> differentiatedErrorFunction(ErrorTag tag)
{
    switch(tag)
    {
        case(ErrorTag::MSE):
            return differentiatedMse<T>;
        case(ErrorTag::CROSS_ENTROPY):
            return differentiatedCrossEntropy<T>;
        case(ErrorTag::NONE):
            throw std::runtime_error{"invalid error tag"};
    }

    return {};
}

}

#endif