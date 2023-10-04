#ifndef NEURAL_NETWORK_FUNCTION_HPP
#define NEURAL_NETWORK_FUNCTION_HPP

#include <cmath>

namespace FUNCTION
{

inline double eluAlpha{1.0};

template<class T>
inline T none(T input)
    {return input;}

template<class T>
inline T elu(T input)
    {return input >= T{0} ? input : std::exp(input) - T{eluAlpha};}

template<class T>
inline T mse(T teacher, T output)
    {return std::pow(teacher - output, T{2}) / T{2};}

template<class T>
inline T differentiatedMse(T teacher, T output)
    {return output - teacher;}

template<class T>
inline T differentiatedElu(T output)
    {return output >= T{0} ? }

}

#endif