#include <stdexcept>
#include <functional>

#include "function.hpp"
#include "layer.hpp"

Layer::Layer(std::size_t column
    , ActivationTag tag)
    : mActivationTag{tag}
    , mInput{1ull
        , column}
    , mOutput{1ull
        , column}
{
}

bool Layer::activate()
{
    std::function<double(double)> activationFunction{FUNCTION::activationFunction<double>(activationTag())};
    
    switch(activationTag())
    {
        case(ActivationTag::SOFTMAX):
            FUNCTION::softmaxSum = 0.0;
            for(std::size_t c{0ull}; c < input().column(); c++)
                FUNCTION::softmaxSum += std::exp(input()[0ull][c]);
            break;
        
        default:
            break;
    }

    for(std::size_t c{0ull}; c < input().column(); c++)
        output()[0ull][c] = activationFunction(input()[0ull][c]);

    return true;
}

void Layer::input(const Matrix<double> &other)
{
    if(input().row() != other.row()
        || input().column() != other.column())
        throw std::runtime_error{"matrix size does not match"};
    
    mInput = other;
}

void Layer::input(Matrix<double> &&other)
{
    if(input().row() != other.row()
        || input().column() != other.column())
        throw std::runtime_error{"matrix size does not match"};
    
    mInput = other;
}