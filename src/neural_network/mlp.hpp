#ifndef NEURAL_NETWORK_MLP_HPP
#define NEURAL_NETWORK_MLP_HPP

#include <vector>
#include <list>
#include <string>

#include "matrix/matrix.hpp"
#include "tag.hpp"

class Layer;
class Weight;
class Bias;

class Mlp
{
public:
    Mlp();
    ~Mlp();

    void addLayer(Layer *layer);
    bool train(std::size_t epochSize
        , std::size_t batchSize
        , ErrorTag errorTag
        , const std::vector<Matrix<double>> &trainingInput
        , const std::vector<Matrix<double>> &trainingOutput
        , const std::vector<Matrix<double>> &validationInput
        , const std::vector<Matrix<double>> &validationOutput
        , const std::vector<Matrix<double>> &testInput
        , const std::vector<Matrix<double>> &testOutput);

private:
    bool randomizeParameter();

    bool trainingError(const std::string &what) const;

    std::list<Layer*> mLayers;
    std::list<Weight*> mWeights;
    std::list<Bias*> mBiases;
};

#endif