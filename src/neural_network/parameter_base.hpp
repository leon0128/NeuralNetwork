#ifndef NEURAL_NETWORK_PARAMETER_BASE_HPP
#define NEURAL_NETWORK_PARAMETER_BASE_HPP

#include <iostream>

#include "matrix/matrix.hpp"
#include "function.hpp"

namespace NEURAL_NETWORK
{

template<class T>
class ParameterBase
{
public:
    friend class Saver;
    friend class Loader;

    ParameterBase(std::size_t row
        , std::size_t column);
    virtual ~ParameterBase();

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

}

#endif