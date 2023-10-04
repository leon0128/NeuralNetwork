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
    std::function<double(double)> activationFunction;

    switch(activationTag())
    {
        case(ActivationTag::NONE):
            activationFunction = FUNCTION::activateNone<double>;
            break;

        case(ActivationTag::ELU):
            activationFunction = FUNCTION::activateElu<double>;
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