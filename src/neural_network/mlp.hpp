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
        , OptimizationTag optimizationTag
        , const std::vector<Matrix<double>> &trainingInput
        , const std::vector<Matrix<double>> &trainingOutput
        , const std::vector<Matrix<double>> &validationInput
        , const std::vector<Matrix<double>> &validationOutput
        , const std::vector<Matrix<double>> &testInput
        , const std::vector<Matrix<double>> &testOutput
        , bool shouldStopEarly = true);

    bool activate(const Matrix<double> &input
        , Matrix<double> &output);

private:
    bool checkValidity(std::size_t epochSize
        , std::size_t batchSize
        , ErrorTag errorTag
        , OptimizationTag optimizationTag
        , const std::vector<Matrix<double>> &trainingInput
        , const std::vector<Matrix<double>> &trainingOutput
        , const std::vector<Matrix<double>> &validationInput
        , const std::vector<Matrix<double>> &validationOutput
        , const std::vector<Matrix<double>> &testInput
        , const std::vector<Matrix<double>> &testOutput) const;
    bool checkValidity(const Matrix<double> &input) const;
    bool randomizeParameter();
    bool propagate(const Matrix<double> &trainingInput);
    bool backpropagate(const Matrix<double> &trainingOutput
        , ErrorTag errorTag
        , std::list<Weight*> &weightGradients
        , std::list<Bias*> &biasGradients);
    bool calculateAverage(std::size_t batchSize
        , std::list<Weight*> &weightGradients
        , std::list<Bias*> &biasGradients);
    bool updateParameter(OptimizationTag optimizationTag
        , std::list<Weight*> &weightGradients
        , std::list<Bias*> &biasGradients);
    double calculateError(const std::vector<Matrix<double>> &inputs
        , const std::vector<Matrix<double>> &outputs
        , ErrorTag errorTag);

    bool trainingError(const std::string &what) const;
    bool activationError(const std::string &what) const;

    std::list<Layer*> mLayers;
    std::list<Weight*> mWeights;
    std::list<Bias*> mBiases;
};

#endif