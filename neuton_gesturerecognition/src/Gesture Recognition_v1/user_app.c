#include <string.h>
#include "user_app.h"

#include "neuton/calculator.h"

static NeuralNet neuralNet = { 0 };
static uint32_t memUsage = 0;

extern const unsigned char model_bin[];
extern const unsigned int model_bin_len;

#if defined(NEUTON_MEMORY_BENCHMARK)
uint32_t _NeutonExtraMemoryUsage()
{
	return memUsage;
}
#endif


inline Err CalculatorOnInit(NeuralNet* neuralNet)
{
	memUsage += sizeof(*neuralNet);
	return CalculatorLoadFromMemory(neuralNet, model_bin, model_bin_len, 0);
}


inline void CalculatorOnFree(NeuralNet* neuralNet)
{

}


inline Err CalculatorOnLoad(NeuralNet* neuralNet)
{
	return ERR_NO_ERROR;
}


inline Err CalculatorOnRun(NeuralNet* neuralNet)
{
	return ERR_NO_ERROR;
}


inline void CalculatorOnInferenceStart(NeuralNet* neuralNet)
{

}


inline void CalculatorOnInferenceEnd(NeuralNet* neuralNet)
{

}

inline void CalculatorOnInferenceResult(NeuralNet* neuralNet, float* result)
{

}

uint8_t model_init()
{
	uint8_t res;
	
	res = CalculatorInit(&neuralNet, NULL);
	
	return (ERR_NO_ERROR == res);
}

float* model_run_inference(float* sample, uint32_t size_in, uint32_t *size_out)
{
	if (!sample || !size_out)
		return NULL;

	if (size_in != neuralNet.inputsDim)
		return NULL;

	*size_out = neuralNet.outputsDim;

	return CalculatorRunInference(&neuralNet, sample);
}
