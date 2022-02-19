#ifndef NEUTON_H
#define NEUTON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Error codes
 */
typedef enum Err_
{
	ERR_NO_ERROR                = 0,
	ERR_OPEN_FILE               = 1,
	ERR_READ_FILE               = 2,
	ERR_BAD_FILE_FORMAT         = 3,
	ERR_INCONSISTENT_DATA       = 4,
	ERR_MEMORY_ALLOCATION       = 5,
	ERR_FEATURE_NOT_SUPPORTED   = 6,
	ERR_BAD_ARGUMENT            = 7

} Err;

/**
 * \brief Task types
 */
typedef enum TaskType_
{
	TASK_MULTICLASS_CLASSIFICATION  = 0,
	TASK_BINARY_CLASSIFICATION      = 1,
	TASK_REGRESSION                 = 2

} TaskType;

/**
 * \brief Model options
 */
typedef enum OptionsBitmask_
{
	BIT_ONE_MAXMIN_FOR_ALL_INPUTS       = 1 << 7,
	BIT_LOG_SCALE_OUT_EXISTS            = 1 << 6,
	BIT_FORCE_INTEGER_CALCULATIONS      = 1 << 5,

} OptionsBitmask;

/**
 * \brief Pointer union
 */
typedef union Pointer_
{
   uint8_t*  u8;
   uint16_t* u16;
   uint32_t* u32;
   uint64_t* u64;
   int8_t*   i8;
   int16_t*  i16;
   int32_t*  i32;
   int64_t*  i64;
   float*    f32;
   void*     raw;

} Pointer;

/**
 * \brief Model structure
 */
typedef struct NeuralNet_
{
	/**
	 * \brief Pointers to connection between neurons
	 */
	Pointer   intLinks;

	/**
	 * \brief Pointers to connections between neurons and model inputs
	 */
	Pointer   extLinks;

	/**
	 * \brief Connection indexes
	 */
	uint16_t* links;

	/**
	 * \brief Connection weights
	 */
	Pointer   weights;

	/**
	 * \brief Buffer for the neurons outs
	 */
	Pointer   accumulators;

	/**
	 * \brief Coefficients of the activation functions
	 */
	Pointer   fncCoeffs;

	/**
	 * \brief Neurons internal connection count
	 */
	uint16_t* intLinksCounters;

	/**
	 * \brief Neurons external connection count
	 */
	uint16_t* extLinksCounters;

	/**
	 * \brief Indexes of output neurons
	 */
	uint16_t* outputLabels;

	/**
	 * \brief Buffer for output data
	 */
	float*    outputBuffer;

	/**
	 * \brief Cached value
	 */
	float     cachedInputsDiff;

	/**
	 * \brief Neurons count
	 */
	uint32_t  neuronsCount;

	/**
	 * \brief Quantisation type
	 */
	uint8_t   quantisation;

	/**
	 * \brief Options: Log scale, Single min/max for inputs
	 */
	uint8_t   options;

	/**
	 * \brief Task Type: Multi Classification, Binary Classification, Regression
	 */
	uint8_t   taskType;

	/**
	 * \brief Flag of the need to reverse data when loading model an meta from files
	 */
	uint8_t   reverseByteOrder;

	/**
	 * \brief Dimension of neural network inputs
	 */
	uint16_t  inputsDim;

	/**
	 * \brief Dimension of neural network outputs
	 */
	uint16_t  outputsDim;

	/**
	 * \brief Dimension of weights array
	 */
	uint32_t  weightDim;

	/**
	 * \brief Neural network maximum inputs
	 */
	float*    inputsMax;

	/**
	 * \brief Neural network minimum inputs
	 */
	float*    inputsMin;

	/**
	 * \brief Neural network maximum outputs
	 */
	float*    outputsMax;

	/**
	 * \brief Neural network minimum outputs
	 */
	float*    outputsMin;

	/**
	 * \brief Logarithmic outputs offset
	 */
	float*    outputsLogOffset;

	/**
	 * \brief User data
	 */
	void*     data;

	/**
	 * \brief Allocated memory block
	 */
	void*     memoryBlock;

} NeuralNet;

/**
 * \brief File descriptor structure
 */
typedef struct NFile_ NFile;

/**
 * \brief Parameters of the loaded dataset
 */
typedef struct Dataset_
{
	NFile*   file;
	uint32_t endDataPos;
	uint32_t sampleDim;
	uint8_t  reverseByteOrder;

} Dataset;


/**
 * \brief Load model using file descriptor
 * \param file - model file
 * \param weightsFile - file with model weights
 * \param model - model of neural network
 * \return error code or 0 on success
 */
extern Err NLoadModel(NFile* file, NeuralNet* model, uint8_t copy);

/**
 * \brief Load model using filename
 * \param fileName - path to model file
 * \param model - model of neural network
 * \return error code or 0 on success
 */
extern Err NLoadModelEx(const char* fileName, NeuralNet* model);

/**
 * \brief Free resources used by model
 * \param model - model of neural network
 */
extern void NFreeModel(NeuralNet* model);

/**
 * \brief Change sample value to the value from the 0.0 - 1.0 range based on the info about
 *        minimums and maximums from the training
 * \param sample - pointer to the buffer with data
 * \param model - pointer to model structure
 */
extern void NNormalizeSample(float* sample, NeuralNet* model);

/**
 * \brief Lead prediction results to the size of training data
 * \param sample - pointer to the buffer with data
 * \param model - pointer to model structure
 */
extern void NDenormalizeResult(float* sample, NeuralNet* model);

/**
 * \brief Run inference
 * \param model - model of neural network
 * \param inputs - vector of input values (size model->inputsDim)
 * \return pointer to buffer with output values (size model->outputsDim)
 */
extern float* NRunInference(NeuralNet* model, float* inputs);

/**
 * \brief Open dataset for line-by-line reading
 * \param file - binary file
 * \param dataset - pointer to the structure for data about the file
 * \return error code or 0 on success
 */
extern Err NOpenDataset(NFile *file, Dataset* dataset);

/**
 * \brief Open dataset for line-by-line reading
 * \param file - name of binary file
 * \param dataset - pointer to the structure for data about the file
 * \return error code or 0 on success
 */
extern Err NOpenDatasetEx(const char* filename, Dataset* dataset);

/**
 * \brief Close dataset, free file descriptor
 * \param dataset - pointer to the structure with dataset info
 */
extern void NCloseDataset(Dataset* dataset);

/**
 * \brief Read one line from file and put the value into the buffer
 * \param dataset - dataset info
 * \param sample - pointer to the buffer for data from dataset
 * \param readSamples - number of read lines; 0 - the end of the file
 * \return error code or 0 on success
 */
extern Err NReadDatasetSample(Dataset* dataset, float* sample, uint32_t *readSamples);

/**
 * \brief Open file by name
 * \param filename - name of the file
 * \param modes - modes (see @fopen for details)
 * \return file descriptor or NULL on failure
 */
extern NFile* NFileOpen(const char* filename, const char* modes);

/**
 * \brief Open file from buffer
 * \param buffer - pointer to file data
 * \param size - size of file
 * \return file descriptor or NULL on failure
 */
extern NFile* NFileFromBuffer(const uint8_t* buffer, uint32_t size);

/**
 * \brief Close file and free file descriptor
 * \param file - file descriptor
 */
extern int32_t NFileClose(NFile* file);

/**
 * \brief Set position in a file
 * \param file - file descriptor
 * \param offset - offset
 * \param whence - point to start from (see @fseek for details)
 * \return returns 0 on success or -1 on failure
 */
extern int32_t NFileSeek(NFile* file, int64_t offset, int32_t whence);

/**
 * \brief Get current position in a file
 * \param file - file descriptor
 * \return current position from begining of a file or -1 on failure
 */
extern int64_t NFilePos(NFile* file);

/**
 * \brief Read data from file
 * \param data - output buffer
 * \param size - size of one item
 * \param count - count of items
 * \param file - file descriptor
 * \return on success, return number of items. If an error occurs, or the end of the file
	   is reached, the return value is a short item count (or zero)
 */
extern uint32_t NFileRead(void* data, uint32_t size, uint32_t count, NFile* file);

/**
 * \brief Allocate memory for an array of count elements of size bytes
 * \param count - elements count
 * \param size - size of element
 * \return pointer to array initialised by zeroes or NULL on failure
 */
extern void* NAlloc(uint32_t count, uint32_t size);

/**
 * \brief Free memory
 * \param ptr - pointer to array aalocated by @NAlloc
 */
extern void NFree(void* ptr);

/**
 * \brief Get current heap usage
 * \return current heap usage
 */
extern uint32_t NBytesAllocated();

/**
 * \brief Get maximum heap usage
 * \return maximum heap usage
 */
extern uint32_t NBytesAllocatedTotal();

#ifdef __cplusplus
}
#endif

#endif  // NEUTON_H
