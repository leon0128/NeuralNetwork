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
    auto &&activationFunction{FUNCTION::activationFunction<double>(activationTag())};

    output() = activationFunction(input());

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