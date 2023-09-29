#include <vector>

#include "matrix/matrix.hpp"
#include "neural_network/mlp.hpp"
#include "neural_network/layer.hpp"

int main(int argc, char **argv)
{
    Mlp mlp;
    mlp.addLayer(new Layer{2ull, ActivationTag::NONE});
    mlp.addLayer(new Layer{2ull, ActivationTag::ELU});
    mlp.addLayer(new Layer{1ull, ActivationTag::ELU});

    std::vector<Matrix<double>> trainingInput;
    std::vector<Matrix<double>> trainingOutput;
    std::vector<Matrix<double>> validationInput;
    std::vector<Matrix<double>> validationOutput;
    std::vector<Matrix<double>> testInput;
    std::vector<Matrix<double>> testOutput;

    Matrix<double> inputA{1ull, 2ull};
    Matrix<double> outputA{1ull, 1ull};
    inputA[0][0] = 0.0;
    inputA[0][1] = 0.0;
    outputA[0][0] = 0.0;
    Matrix<double> inputB{1ull, 2ull};
    Matrix<double> outputB{1ull, 1ull};
    inputB[0][0] = 0.0;
    inputB[0][1] = 1.0;
    outputB[0][0] = 1.0;
    Matrix<double> inputC{1ull, 2ull};
    Matrix<double> outputC{1ull, 1ull};
    inputC[0][0] = 1.0;
    inputC[0][1] = 0.0;
    outputC[0][0] = 1.0;
    Matrix<double> inputD{1ull, 2ull};
    Matrix<double> outputD{1ull, 1ull};
    inputD[0][0] = 1.0;
    inputD[0][1] = 1.0;
    outputD[0][0] = 0.0;

    trainingInput.push_back(inputA);
    trainingInput.push_back(inputB);
    trainingInput.push_back(inputC);
    trainingInput.push_back(inputD);
    trainingOutput.push_back(outputA);
    trainingOutput.push_back(outputB);
    trainingOutput.push_back(outputC);
    trainingOutput.push_back(outputD);

    validationInput = trainingInput;
    validationOutput = trainingOutput;

    testInput = trainingInput;
    testOutput = trainingOutput;
 
    mlp.train(10ull
        , 1ull
        , ErrorTag::MSE
        , trainingInput
        , trainingOutput
        , validationInput
        , validationOutput
        , testInput
        , testOutput);

    return 0;
}