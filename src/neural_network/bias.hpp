#ifndef NEURAL_NETWORK_BIAS_HPP
#define NEURAL_NETWORK_BIAS_HPP

#include "parameter_base.hpp"

class Bias : public ParameterBase
{
public:
    Bias(std::size_t column);
};

#endif