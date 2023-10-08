#ifndef NEURAL_NETWORK_TAG_HPP
#define NEURAL_NETWORK_TAG_HPP

enum class ErrorTag
{
    NONE
    , MSE
    , CROSS_ENTROPY
};

enum class ActivationTag
{
    NONE
    , ELU
    , SIGMOID
    , RELU
    , SOFTMAX
};

enum class OptimizationTag
{
    NONE
    , ADAM
};

#endif