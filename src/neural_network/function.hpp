#ifndef NEURAL_NETWORK_FUNCTION_HPP
#define NEURAL_NETWORK_FUNCTION_HPP

#include <cmath>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <limits>

#include "matrix/matrix.hpp"
#include "tag.hpp"

namespace NEURAL_NETWORK
{

namespace FUNCTION
{

inline const double eluAlpha{1.0};

inline const double optimizationNoneLearningRate{0.01};

inline const double adamLearningRate{0.001};
inline const double adamBeta1{0.9};
inline const double adamBeta2{0.999};
inline const double adamEpsilon{1.0e-7};

template<class T>
inline Matrix<T> activationNone(const Matrix<T> &input)
    {return input;}

template<class T>
inline Matrix<T> elu(const Matrix<T> &input)
    {return input.apply([](T t){return t >= static_cast<T>(0) ? t : std::exp(t) - static_cast<T>(eluAlpha);});}

template<class T>
inline Matrix<T> sigmoid(const Matrix<T> &input)
    {return input.apply([](T t){return static_cast<T>(1) / (static_cast<T>(1) + std::exp(static_cast<T>(-1) * t));});}

template<class T>
inline Matrix<T> relu(const Matrix<T> &input)
    {return input.apply([](T t){return t >= static_cast<T>(0) ? t : static_cast<T>(0);});}

template<class T>
inline Matrix<T> softmax(const Matrix<T> &input)
{
    Matrix output{exp(input)};
    return output /= output.sum();
}

template<class T>
inline T mse(const Matrix<T> &teacher, const Matrix<T> &output)
    {return pow(teacher - output, static_cast<T>(2)).sum() / static_cast<T>(2);}

template<class T>
inline T binaryCrossEntropy(const Matrix<T> &teacher, const Matrix<T> &output)
    {return -((teacher * log(output) + (static_cast<T>(1) - teacher) * log(static_cast<T>(1) - output)).sum());}

template<class T>
inline T categoricalCrossEntropy(const Matrix<T> &teacher, const Matrix<T> &output)
    {return -((teacher * log(output)).sum());}

template<class T>
inline Matrix<T> derivativeActivationNone(const Matrix<T> &output)
    {return Matrix<T>{output.row(), output.column(), static_cast<T>(1)};}

template<class T>
inline Matrix<T> derivativeElu(const Matrix<T> &output)
    {return output.apply([](T t){return t >= static_cast<T>(0) ? static_cast<T>(1) : t + static_cast<T>(eluAlpha);});}

template<class T>
inline Matrix<T> derivativeSigmoid(const Matrix<T> &output)
    {return output * (static_cast<T>(1) - output);}

template<class T>
inline Matrix<T> derivativeRelu(const Matrix<T> &output)
    {return output.apply([](T t){return t >= static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);});}

template<class T>
inline Matrix<T> derivativeSoftmax(const Matrix<T> &output)
{
    Matrix<T> derivative{output.column(), output.column()};
    for(std::size_t r{0ull}; r < derivative.row(); r++)
        for(std::size_t c{0ull}; c < derivative.column(); c++)
            derivative(r, c)
                = r == c
                    ? output(0, r) * (static_cast<T>(1) - output(0, c))
                    : -(output(0, r) * output(0, c));
    return derivative;
}

template<class T>
inline Matrix<T> derivativeMse(const Matrix<T> &teacher, const Matrix<T> &output)
    {return output - teacher;}

template<class T>
inline Matrix<T> derivativeBinaryCrossEntropy(const Matrix<T> &teacher, const Matrix<T> &output)
    {return (static_cast<T>(1) - teacher) / (static_cast<T>(1) - output) - teacher / output;}

template<class T>
inline Matrix<T> derivativeCategoricalCrossEntropy(const Matrix<T> &teacher, const Matrix<T> &output)
    {return -(teacher / output);}

template<class T>
inline Matrix<T> optimizationNone(const Matrix<T> &prev, const Matrix<T> &gradient)
    {return prev - gradient * static_cast<T>(optimizationNoneLearningRate);}

template<class T>
inline Matrix<T> adam(const Matrix<T> &prev
    , const Matrix<T> &gradient
    , Matrix<T> &adamM
    , Matrix<T> &adamV)
{
    adamM = static_cast<T>(adamBeta1) * adamM + (static_cast<T>(1) - static_cast<T>(adamBeta1)) * gradient;
    adamV = static_cast<T>(adamBeta2) * adamV + (static_cast<T>(1) - static_cast<T>(adamBeta2)) * pow(gradient, static_cast<T>(2));
    return prev - (static_cast<T>(adamLearningRate) * adamM / (static_cast<T>(1) - static_cast<T>(adamBeta1)) / (sqrt(adamV / (static_cast<T>(1) - static_cast<T>(adamBeta2))) + static_cast<T>(adamEpsilon)));
}

template<class T>
inline std::function<Matrix<T>(const Matrix<T>&)> activationFunction(ActivationTag tag)
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
inline std::function<T(const Matrix<T>&, const Matrix<T>&)> errorFunction(ErrorTag tag)
{
    switch(tag)
    {
        case(ErrorTag::MSE):
            return mse<T>;
        case(ErrorTag::BINARY_CROSS_ENTROPY):
            return binaryCrossEntropy<T>;
        case(ErrorTag::CATEGORICAL_CROSS_ENTROPY):
            return categoricalCrossEntropy<T>;
        case(ErrorTag::NONE):
            throw std::runtime_error{"invalid error tag"};
    }

    return {};
}

template<class T>
inline std::function<Matrix<T>(const Matrix<T>&)> derivativeActivationFunction(ActivationTag tag)
{
    switch(tag)
    {
        case(ActivationTag::NONE):
            return derivativeActivationNone<T>;
        case(ActivationTag::ELU):
            return derivativeElu<T>;
        case(ActivationTag::SIGMOID):
            return derivativeSigmoid<T>;
        case(ActivationTag::RELU):
            return derivativeRelu<T>;
        case(ActivationTag::SOFTMAX):
            return derivativeSoftmax<T>;
    }

    return {};
}

template<class T>
inline std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)> derivativeErrorFunction(ErrorTag tag)
{
    switch(tag)
    {
        case(ErrorTag::MSE):
            return derivativeMse<T>;
        case(ErrorTag::BINARY_CROSS_ENTROPY):
            return derivativeBinaryCrossEntropy<T>;
        case(ErrorTag::CATEGORICAL_CROSS_ENTROPY):
            return derivativeCategoricalCrossEntropy<T>;
        case(ErrorTag::NONE):
            throw std::runtime_error{"invalid error tag"};
    }

    return {};
}

}

}

#endif