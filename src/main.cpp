#include <vector>
#include <iostream>

#include "matrix/matrix.hpp"
#include "neural_network/neural_network.hpp"

int main(int argc, char **argv)
{
    using namespace NEURAL_NETWORK;

    NeuralNetwork<double> mlp;
    mlp.addLayer(new Layer<double>{2ull, ActivationTag::NONE});
    mlp.addLayer(new Layer<double>{32ull, ActivationTag::ELU, 1.0});
    mlp.addLayer(new Layer<double>{32ull, ActivationTag::ELU, 1.0});
    mlp.addLayer(new Layer<double>{32ull, ActivationTag::ELU, 1.0});
    mlp.addLayer(new Layer<double>{2ull, ActivationTag::SOFTMAX});

    std::vector<Matrix<double>> trainingInput;
    std::vector<Matrix<double>> trainingOutput;
    std::vector<Matrix<double>> validationInput;
    std::vector<Matrix<double>> validationOutput;
    std::vector<Matrix<double>> testInput;
    std::vector<Matrix<double>> testOutput;

    Matrix<double> inputA{1ull, 2ull};
    Matrix<double> outputA{1ull, 2ull};
    inputA[0][0] = 0.0;
    inputA[0][1] = 0.0;
    outputA[0][0] = 1.0;
    outputA[0][1] = 0.0;
    Matrix<double> inputB{1ull, 2ull};
    Matrix<double> outputB{1ull, 2ull};
    inputB[0][0] = 0.0;
    inputB[0][1] = 1.0;
    outputB[0][0] = 0.0;
    outputB[0][1] = 1.0;
    Matrix<double> inputC{1ull, 2ull};
    Matrix<double> outputC{1ull, 2ull};
    inputC[0][0] = 1.0;
    inputC[0][1] = 0.0;
    outputC[0][0] = 0.0;
    outputC[0][1] = 1.0;
    Matrix<double> inputD{1ull, 2ull};
    Matrix<double> outputD{1ull, 2ull};
    inputD[0][0] = 1.0;
    inputD[0][1] = 1.0;
    outputD[0][0] = 1.0;
    outputD[0][1] = 0.0;

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
 
    mlp.train(100'000ull
        , 4ull
        , ErrorTag::CATEGORICAL_CROSS_ENTROPY
        , OptimizationTag::ADAM
        , trainingInput
        , trainingOutput
        , validationInput
        , validationOutput
        , testInput
        , testOutput
        , 100'000);

    Matrix<double> result;
    mlp.predict(inputA, result);
    std::cout << inputA << ": " << result << std::endl;
    mlp.predict(inputB, result);
    std::cout << inputB << ": " << result << std::endl;
    mlp.predict(inputC, result);
    std::cout << inputC << ": " << result << std::endl;
    mlp.predict(inputD, result);
    std::cout << inputD << ": " << result << std::endl;

    mlp.save("test.out");
    NeuralNetwork<double> newMlp;
    newMlp.load("test.out");
    newMlp.predict(inputA, result);
    std::cout << inputA << ": " << result << std::endl;
    newMlp.predict(inputB, result);
    std::cout << inputB << ": " << result << std::endl;
    newMlp.predict(inputC, result);
    std::cout << inputC << ": " << result << std::endl;
    newMlp.predict(inputD, result);
    std::cout << inputD << ": " << result << std::endl;

    return 0;
}