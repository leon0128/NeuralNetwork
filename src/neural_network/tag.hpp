#ifndef NEURAL_NETWORK_TAG_HPP
#define NEURAL_NETWORK_TAG_HPP

enum class ErrorTag
{
    NONE
    , MSE
};

enum class ActivationTag
{
    NONE
    , ELU
};

enum class OptimizationTag
{
    NONE
    , ADAM
};

#endif