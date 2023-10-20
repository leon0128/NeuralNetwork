#ifndef NEURAL_NETWORK_WEIGHT_HPP
#define NEURAL_NETWORK_WEIGHT_HPP

#include "parameter_base.hpp"

namespace NEURAL_NETWORK
{

template<class T>
class Weight : public ParameterBase<T>
{
public:
    Weight(std::size_t row
        , std::size_t column);
    Weight(const ParameterBase<T>&);
    Weight(ParameterBase<T>&&);
};

template<class T>
Weight<T>::Weight(std::size_t row
    , std::size_t column)
    : ParameterBase<T>{row, column}
{
}

template<class T>
Weight<T>::Weight(const ParameterBase<T> &pb)
    : ParameterBase<T>{pb}
{
}

template<class T>
Weight<T>::Weight(ParameterBase<T> &&pb)
    : ParameterBase<T>{pb}
{
}

}

#endif