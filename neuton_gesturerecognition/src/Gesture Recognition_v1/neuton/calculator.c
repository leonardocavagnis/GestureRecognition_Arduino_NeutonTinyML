#include <string.h>

#include "calculator.h"
#include "neuton.h"


inline Err CalculatorInit(NeuralNet* neuralNet, void* userData)
{
	if (!neuralNet)
		return ERR_BAD_ARGUMENT;

	memset(neuralNet, 0, sizeof(NeuralNet));
	neuralNet->data = userData;

	return CalculatorOnInit(neuralNet);
}


inline void CalculatorFree(NeuralNet* neuralNet)
{
	if (!neuralNet)
		return;

	CalculatorOnFree(neuralNet);

	NFreeModel(neuralNet);
}


inline Err CalculatorLoadFromMemory(NeuralNet* neuralNet, const void* model, uint32_t size, uint8_t copy)
{
	if (!model || !size)
		return ERR_BAD_ARGUMENT;

	Err err = NLoadModel(NFileFromBuffer(model, size), neuralNet, copy);

	if (ERR_NO_ERROR == err)
		err = CalculatorOnLoad(neuralNet);

	return err;
}


inline Err CalculatorLoadFromFile(NeuralNet* neuralNet, const char* fileName)
{
	Err err = NLoadModelEx(fileName, neuralNet);

	if (ERR_NO_ERROR == err)
		err = CalculatorOnLoad(neuralNet);

	return err;
}


inline Err CalculatorRunApplication(NeuralNet* neuralNet)
{
	if (!neuralNet)
		return ERR_BAD_ARGUMENT;

	return CalculatorOnRun(neuralNet);
}


inline float* CalculatorRunInference(NeuralNet* neuralNet, float* inputs)
{
	if (!neuralNet || !inputs)
		return NULL;

	/**
	 * Normalize sample values in the buffer
	 */
	NNormalizeSample(inputs, neuralNet);

	CalculatorOnInferenceStart(neuralNet);

	/**
	 * Get result of prediction
	 */
	float* result = NRunInference(neuralNet, inputs);
	if (!result)
		return result;

	CalculatorOnInferenceEnd(neuralNet);

	/**
	 * Restore result values
	 */
	NDenormalizeResult(result, neuralNet);

	CalculatorOnInferenceResult(neuralNet, result);

	return result;
}
