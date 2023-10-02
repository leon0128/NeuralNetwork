#include <functional>

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
            activationFunction = &activateNone;
            break;

        case(ActivationTag::ELU):
        {
            if()
            break;
        }
    }

    return true;
}