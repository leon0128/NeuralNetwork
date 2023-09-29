#include "parameter_base.hpp"

ParameterBase::ParameterBase(std::size_t row
    , std::size_t column)
    : mData{row, column}
{
}

ParameterBase::~ParameterBase()
{
}