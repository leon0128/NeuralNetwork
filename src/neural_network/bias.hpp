#ifndef NEURAL_NETWORK_BIAS_HPP
#define NEURAL_NETWORK_BIAS_HPP

#include "parameter_base.hpp"

namespace NEURAL_NETWORK
{

template<class T>
class Bias : public ParameterBase<T>
{
public:
    Bias(std::size_t column);
    Bias(const ParameterBase<T>&);
    Bias(ParameterBase<T>&&);
};

template<class T>
Bias<T>::Bias(std::size_t column)
    : ParameterBase<T>{1ull, column}
{
}

template<class T>
Bias<T>::Bias(const ParameterBase<T> &pb)
    : ParameterBase<T>{pb}
{
}

template<class T>
Bias<T>::Bias(ParameterBase<T> &&pb)
    : ParameterBase<T>{pb}
{
}

}

#endif