#ifndef NEURAL_NETWORK_TAG_HPP
#define NEURAL_NETWORK_TAG_HPP

namespace NEURAL_NETWORK
{

enum class ErrorTag
{
    NONE
    , MSE
    , BINARY_CROSS_ENTROPY
    , CATEGORICAL_CROSS_ENTROPY
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

}

#endif