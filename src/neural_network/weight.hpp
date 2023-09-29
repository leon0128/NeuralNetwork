#ifndef NEURAL_NETWORK_WEIGHT_HPP
#define NEURAL_NETWORK_WEIGHT_HPP

#include "parameter_base.hpp"

class Weight : public ParameterBase
{
public:
    Weight(std::size_t row
        , std::size_t column);
};

#endif