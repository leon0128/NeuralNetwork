#ifndef NEURAL_NETWORK_CONCURRENCY_HPP
#define NEURAL_NETWORK_CONCURRENCY_HPP

#include <functional>

namespace NEURAL_NETWORK
{

namespace CONCURRENCY
{

// function's return type should be bool.
// if funcion returns true, execution is terminated.
// at the end of execution, running function is processed to the end.
void execute(const std::function<bool()> &function
    , std::size_t concurrencySize);

}

}

#endif