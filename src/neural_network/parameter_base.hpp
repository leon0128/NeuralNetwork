#ifndef NEURAL_NETWORK_PARAMETER_BASE_HPP
#define NEURAL_NETWORK_PARAMETER_BASE_HPP

#include "matrix/matrix.hpp"

template<class T>
class ParameterBase
{
public:
    ParameterBase(std::size_t row
        , std::size_t column);
    virtual ~ParameterBase() = 0;

    auto &data()
        {return mData;}
    const auto &data() const
        {return mData;}

protected:
    Matrix<T> mData;
};

template<class T>
ParameterBase<T>::ParameterBase(std::size_t row
    , std::size_t column)
    : mData{row, column}
{
}

template<class T>
ParameterBase<T>::~ParameterBase()
{
}

#endif