#ifndef NEURAL_NETWORK_RANDOM_HPP
#define NEURAL_NETWORK_RANDOM_HPP

#include <random>

namespace NEURAL_NETWORK
{

template<class E = std::mt19937_64
    , class S = std::random_device>
class Random
{
public:
    using Engine = E;
    using Seed = S;
    using ResultType = typename Engine::result_type;

    inline Random()
        : mEngine{S{}()}{}

    // return: [0, ResultType::max]
    inline ResultType operator()()
        {return mEngine();}
    // return: [0, max)
    inline ResultType operator()(ResultType max)
        {return mEngine() % max;}
    // return: [min, max)
    inline ResultType operator()(ResultType min, ResultType max)
        {return mEngine() % (max - min) + min;}

    // return: [0, T::max]
    template<class T = ResultType>
    inline T random()
        {return static_cast<T>(mEngine());}
    // return: [0, max)
    template<class T = ResultType>
    inline T random(T max)
        {return random<T>() % max;}
    // return: [min, max)
    template<class T = ResultType>
    inline T random(T min, T max)
        {return random<T>() % (max - min) + min;}

    // return: [0.0, 1.0]
    template<class FloatingType = double>
    inline FloatingType floating()
        {return static_cast<FloatingType>(random()) / static_cast<FloatingType>(Engine::max());}

    // engine
    inline Engine &engine() noexcept
        {return mEngine;}

private:
    Engine mEngine;
};

inline static Random RANDOM{};

}

#endif