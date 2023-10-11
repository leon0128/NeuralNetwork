#ifndef NEURAL_NETWORK_FUNCTION_HPP
#define NEURAL_NETWORK_FUNCTION_HPP

#include <cmath>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <limits>

#include "matrix/matrix.hpp"
#include "tag.hpp"

namespace FUNCTION
{

inline const double eluAlpha{1.0};

inline const double optimizationNoneLearningRate{0.01};

inline const double adamLearningRate{0.001};
inline const double adamBeta1{0.9};
inline const double adamBeta2{0.999};
inline const double adamEpsilon{1.0e-7};
inline Matrix<double> adamM{};
inline Matrix<double> adamV{};

template<class T>
inline Matrix<T> activationNone(const Matrix<T> &input)
{
    return input;
}

template<class T>
inline Matrix<T> elu(const Matrix<T> &input)
{
    Matrix<T> output{input};
    output.apply([](T in){return in >= T{0} ? in : std::exp(in) - T{eluAlpha};});
    return output;
}

template<class T>
inline Matrix<T> sigmoid(const Matrix<T> &input)
{
    Matrix<T> output{input};
    output.apply([](T in){return T{1} / (T{1} + std::exp(T{-1} * in));});
    return output;
}

template<class T>
inline Matrix<T> relu(const Matrix<T> &input)
{
    Matrix<T> output{input};
    output.apply([](T in){return in >= T{0} ? in : T{0};});
    return output;
}

template<class T>
inline Matrix<T> softmax(const Matrix<T> &input)
{
    Matrix<T> output{input};
    T sum{0};
    output.apply([&](T in){double out{std::exp(in)}; sum += out; return out;});
    output.apply([&](T in){return in / sum;});
    return output;
}

template<class T>
inline T mse(const Matrix<T> &teacher, const Matrix<T> &output)
{
    T error{0};
    for(std::size_t r{0ull}; r < teacher.row(); r++)
        for(std::size_t c{0ull}; c < teacher.column(); c++)
            error += std::pow(teacher[r][c] - output[r][c], T{2});
    return error / T{2};
}

template<class T>
inline T binaryCrossEntropy(const Matrix<T> &teacher, const Matrix<T> &output)
{
    T error{0};
    for(std::size_t r{0ull}; r < teacher.row(); r++)
        for(std::size_t c{0ull}; c < teacher.column(); c++)
            error += teacher[r][c] * std::log(output[r][c])
                + (T{1} - teacher[r][c]) * std::log(T{1} - output[r][c]);
    return error * T{-1};
}

template<class T>
inline T categoricalCrossEntropy(const Matrix<T> &teacher, const Matrix<T> &output)
{
    T error{0};
    for(std::size_t r{0ull}; r < teacher.row(); r++)
        for(std::size_t c{0ull}; c < teacher.column(); c++)
            error += teacher[r][c] * std::log(output[r][c]);
    return error * T{-1};
}

template<class T>
inline Matrix<T> derivativeActivationNone(const Matrix<T> &output)
{
    Matrix<T> derivative{output.row(), output.column()};
    derivative.apply([](T in){return T{1};});
    return derivative;
}

template<class T>
inline Matrix<T> derivativeElu(const Matrix<T> &output)
{
    Matrix<T> derivative{output};
    derivative.apply([](T in){return in >= T{0} ? T{1} : in + eluAlpha;});
    return derivative;
}

template<class T>
inline Matrix<T> derivativeSigmoid(const Matrix<T> &output)
{
    Matrix<T> derivative{output};
    derivative.apply([](T in){return in * (T{1} - in);});
    return derivative;
}

template<class T>
inline Matrix<T> derivativeRelu(const Matrix<T> &output)
{
    Matrix<T> derivative{output};
    derivative.apply([](T in){return in >= T{0} ? T{1} : T{0};});
    return derivative;
}

template<class T>
inline Matrix<T> derivativeSoftmax(const Matrix<T> &output)
{
    Matrix<T> derivative{output.column(), output.column()};
    for(std::size_t r{0ull}; r < derivative.row(); r++)
        for(std::size_t c{0ull}; c < derivative.column(); c++)
            derivative[r][c]
                = r == c
                    ? output[0ull][r] * (T{1} - output[0ull][c])
                    : T{-1} * output[0ull][r] * output[0ull][c];
    return derivative;
}

template<class T>
inline Matrix<T> derivativeMse(const Matrix<T> &teacher, const Matrix<T> &output)
{
    return output - teacher;
}

template<class T>
inline Matrix<T> derivativeBinaryCrossEntropy(const Matrix<T> &teacher, const Matrix<T> &output)
{
    Matrix<T> derivative{teacher.row(), teacher.column()};
    for(std::size_t r{0ull}; r < derivative.row(); r++)
        for(std::size_t c{0ull}; c < derivative.column(); c++)
            derivative[r][c] = (T{1} - teacher[r][c]) / (T{1} - output[r][c])
                - teacher[r][c] / output[r][c];
    return derivative;            
}

template<class T>
inline Matrix<T> derivativeCategoricalCrossEntropy(const Matrix<T> &teacher, const Matrix<T> &output)
{
    Matrix<T> derivative{teacher.row(), teacher.column()};
    for(std::size_t r{0ull}; r < derivative.row(); r++)
        for(std::size_t c{0ull}; c < derivative.column(); c++)
            derivative[r][c] = T{-1} * teacher[r][c] / output[r][c];
    return derivative;
}

template<class T>
inline Matrix<T> optimizationNone(const Matrix<T> &prev, const Matrix<T> &gradient)
{
    Matrix<T> rhs{gradient};
    rhs.apply([](T in){return in * optimizationNoneLearningRate;});
    return prev - rhs;
}

template<class T>
inline Matrix<T> adam(const Matrix<T> &prev, const Matrix<T> &gradient)
{
    Matrix<T> parameter{prev.row(), prev.column()};
    for(std::size_t r{0ull}; r < gradient.row(); r++)
    {
        for(std::size_t c{0ull}; c < gradient.column(); c++)
        {
            adamM[r][c] = adamBeta1 * adamM[r][c] + (T{1} - adamBeta1) * gradient[r][c];
            adamV[r][c] = adamBeta2 * adamV[r][c] + (T{1} - adamBeta2) * std::pow(gradient[r][c], T{2});
            double mHat{adamM[r][c] / (T{1} - adamBeta1)};
            double vHat{adamV[r][c] / (T{1} - adamBeta2)};
            parameter[r][c] = prev[r][c] - (adamLearningRate * mHat / (std::sqrt(vHat) + adamEpsilon));
        }
    }
    return parameter;
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
inline std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)> optimizationFunction(OptimizationTag tag)
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

#endif