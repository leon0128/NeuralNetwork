#ifndef NEURAL_NETWORK_FUNCTION_HPP
#define NEURAL_NETWORK_FUNCTION_HPP

#include <cmath>

namespace FUNCTION
{

template<class T>
inline T activateNone(T input)
    {return input;}

template<class T>
inline T activateElu(T input)
    {return input >= T{0} ? input : std::exp(input) - T{1};}

}

#endif