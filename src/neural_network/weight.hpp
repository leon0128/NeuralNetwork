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
};

template<class T>
Weight<T>::Weight(std::size_t row
    , std::size_t column)
    : ParameterBase<T>{row, column}
{
}

}

#endif