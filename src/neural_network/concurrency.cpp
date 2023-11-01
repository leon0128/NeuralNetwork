#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <numeric>
#include <iostream>

#include "concurrency.hpp"

namespace NEURAL_NETWORK
{

namespace CONCURRENCY
{

void execute(const std::function<bool(std::size_t)> &function
    , std::size_t concurrencySize)
{
    if(concurrencySize == 0ull)
        return;

    // mutex and cv are used to availableIndices
    std::mutex mutex;
    std::condition_variable cv;

    std::deque<std::future<bool>> futures{concurrencySize};
    std::deque<std::size_t> availableIndices(concurrencySize);
    
    std::iota(availableIndices.begin()
        , availableIndices.end()
        , 0ull);

    // index: futures's index
    auto &&functionWrapper{[&](std::size_t index)
        -> bool
        {
            bool result{function(index)};
        
            std::unique_lock lock{mutex};
            availableIndices.push_back(index);
            cv.notify_all();
        
            return result;
        }};

    while(true)
    {
        std::unique_lock lock{mutex};

        if(availableIndices.empty())
            cv.wait(lock, [&]{return !availableIndices.empty();});
        else
        {
            std::size_t index{availableIndices.front()};
            availableIndices.pop_front();

            if(futures.at(index).valid())
            {
                if(!futures.at(index).get())
                    break;
            }
            
            futures.at(index)
                = std::async(std::launch::async
                    , functionWrapper
                    , index);
        }
    }

    for(auto &&future : futures)
    {
        if(future.valid())
            future.get();
    }
}

}

}