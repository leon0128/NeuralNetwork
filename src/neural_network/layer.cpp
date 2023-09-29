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