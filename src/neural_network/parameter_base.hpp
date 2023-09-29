#ifndef NEURAL_NETWORK_PARAMETER_BASE_HPP
#define NEURAL_NETWORK_PARAMETER_BASE_HPP

#include "matrix/matrix.hpp"

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
    Matrix<double> mData;
};

#endif