#ifndef CALCULATOR_H
#define CALCULATOR_H

#include "neuton.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Init NeuralNet structure
 * @param neuralNet - pointer to NeuralNet structure
 * @param userData - pointer to user data, can be read from neuralNet.data
 * @return Error code
 */
Err CalculatorInit(NeuralNet* neuralNet, void* userData);

/**
 * @brief Deinit NeuralNet sructure
 * @param neuralNet
 */
void CalculatorFree(NeuralNet* neuralNet);

/**
 * @brief Load model from memory
 * @param neuralNet - pointer to NeuralNet structure
 * @param model - pointer to model data
 * @param size - size of model data
 * @return Error code
 */
Err CalculatorLoadFromMemory(NeuralNet* neuralNet, const void* model, uint32_t size, uint8_t copy);

/**
 * @brief Load model from file
 * @param neuralNet - pointer to NeuralNet structure
 * @param fileName - path to model file
 * @return Error code
 */
Err CalculatorLoadFromFile(NeuralNet* neuralNet, const char* fileName);

/**
 * @brief Run user application
 * @param neuralNet - pointer to NeuralNet structure
 * @return Error code
 */
Err CalculatorRunApplication(NeuralNet* neuralNet);

/**
 * @brief Run inference
 * @param neuralNet - pointer to NeuralNet structure
 * @param inputs - vector of input values (neuralNet->inputsDim elements)
 * @return pointer to buffer with output values (neuralNet->outputsDim elements)
 */
float* CalculatorRunInference(NeuralNet* neuralNet, float* inputs);

/**
 *
 * Callback functions
 *
 */

/**
 * @brief Callback function for additional initialisation
 * @details Called from @calculatorInit
 * @param neuralNet - pointer to NeuralNet structure
 * @return Error code
 */
Err CalculatorOnInit(NeuralNet* neuralNet);

/**
 * @brief Callback function for additional deinitialisation
 * @details Called from @calculatorFree before destroying @neuralNet structure
 * @param neuralNet - pointer to NeuralNet structure
 */
void CalculatorOnFree(NeuralNet* neuralNet);

/**
 * @brief Callback function for actions after model is loaded
 * @details Called from @calculatorLoadFromMemory and @calculatorLoadFromFile
 * @param neuralNet - pointer to NeuralNet structure
 * @return Error code
 */
Err CalculatorOnLoad(NeuralNet* neuralNet);

/**
 * @brief Callback function for running user application
 * @details Called from @calculatorRunApplication
 * @param neuralNet - pointer to NeuralNet structure
 * @return Error code
 */
Err CalculatorOnRun(NeuralNet* neuralNet);

/**
 * @brief Callback function called berofe inference, after inputs normalisation
 * @details Called from @calculatorRunInference
 * @param neuralNet - pointer to NeuralNet structure
 */
void CalculatorOnInferenceStart(NeuralNet* neuralNet);


/**
 * @brief Callback function called after inference, before result denormalisation
 * @details Called from @calculatorRunInference
 * @param neuralNet - pointer to NeuralNet structure
 */
void CalculatorOnInferenceEnd(NeuralNet* neuralNet);


/**
 * @brief Callback function called after inference is finished
 * @details Called from @calculatorRunInference
 * @param neuralNet - pointer to NeuralNet structure
 * @param result - pointer to output buffer
 */
void CalculatorOnInferenceResult(NeuralNet* neuralNet, float* result);


#ifdef __cplusplus
}
#endif

#endif // CALCULATOR_H
