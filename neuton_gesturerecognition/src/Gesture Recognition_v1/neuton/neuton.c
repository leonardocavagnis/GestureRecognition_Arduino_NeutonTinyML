#include "neuton.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>


#if defined(NEUTON_USE_STDIO)
#include <stdio.h>
#else
#define SEEK_SET	0	/* Seek from beginning of file.  */
#define SEEK_CUR	1	/* Seek from current position.  */
#define SEEK_END	2	/* Seek from end of file.  */
#endif


#define MAX_INPUT_FLOAT			0.9999999f
#define MAX_INPUT_DOUBLE		0.999999999999999

#define KSHIFT_2				2
#define KSHIFT_10				10


#if !defined(NEUTON_Q32_SUPPORT)
#define NEUTON_Q32_SUPPORT		1
#endif
#if !defined(NEUTON_Q16_SUPPORT)
#define NEUTON_Q16_SUPPORT		1
#endif


#if defined(NEUTON_MEMORY_BENCHMARK)
#if !defined(NEUTON_MEMORY_BENCHMARK_SIZE)
#define NEUTON_MEMORY_BENCHMARK_SIZE 4
#endif

#include <assert.h>
#include <stdio.h>


typedef struct MemoryChunk_
{
	void*    ptr;
	uint32_t size;

}
MemoryChunk;


static MemoryChunk chunks[NEUTON_MEMORY_BENCHMARK_SIZE] = { 0 };

static uint32_t memAllocated = 0;
static uint32_t memAllocatedMax = 0;
static uint8_t  allocs = 0;
static uint8_t  allocsMax = 0;

extern uint32_t _NeutonExtraMemoryUsage();
#endif


static const uint8_t pointerTypeSize = sizeof(void*);


/**
 * \brief File types
 */
typedef enum FileType_
{
	TYPE_UNKNOWN = 0,
	TYPE_DATASET = 1,
	TYPE_MODEL   = 5,

} FileType;

/**
 * \brief Common header of binary files
 */
typedef struct __attribute__((packed)) BinHeader_
{
	char     nb[2];
	uint8_t  type;
	uint8_t  version;
	uint16_t bom;

} BinHeader;

/**
 * \brief Header of meta file
 */
typedef struct __attribute__((packed)) MetaInfo_
{
	uint8_t  options;
	uint8_t  taskType;
	uint16_t inputsDim;
	uint16_t outputsDim;
	uint8_t  quantisation;
	uint8_t  reserved;
	uint16_t neuronsCount;

} MetaInfo;

struct NFile_
{
#if defined(NEUTON_USE_STDIO)
	FILE* desc;
#endif

	uint8_t* data;
	uint32_t size;
	uint32_t pos;
};


NFile *NFileOpen(const char *filename, const char *modes)
{
#if defined(NEUTON_USE_STDIO)
	FILE* desc = fopen(filename, modes);
	if (!desc)
		return NULL;

	NFile* file = NAlloc(1, sizeof(struct NFile_));
	if (!file)
	{
		fclose(desc);
		return NULL;
	}

	file->desc = desc;

	NFileSeek(file, 0, SEEK_END);
	file->size = NFilePos(file);
	NFileSeek(file, 0, SEEK_SET);

	return file;
#else
	return NULL;
#endif
}


NFile *NFileFromBuffer(const uint8_t* buffer, uint32_t size)
{
	if (!buffer || !size)
		return NULL;

	NFile* file = NAlloc(1, sizeof(struct NFile_));
	if (!file)
		return NULL;

	file->data = (uint8_t*) buffer;
	file->size = size;
	file->pos = 0;

	return file;
}


int32_t NFileClose(NFile *file)
{
	int32_t res = 0;

	if (!file)
		return res;

#if defined(NEUTON_USE_STDIO)
	if (file->desc)
		res = fclose(file->desc);
#endif

	memset(file, 0, sizeof(struct NFile_));
	NFree(file);

	return res;
}


int32_t NFileSeek(NFile *file, int64_t offset, int32_t whence)
{
#if defined(NEUTON_USE_STDIO)
	if (file->desc)
		return fseek(file->desc, offset, whence);
#endif

	int32_t res = -1;
	int64_t newPos = -1;

	switch (whence)
	{
	case SEEK_SET:
		newPos = offset;
		break;

	case SEEK_CUR:
		newPos = (int64_t) file->pos + offset;
		break;

	case SEEK_END:
		newPos = (int64_t) file->size + offset;
		break;

	default:;
	}

	if (newPos >= 0 && newPos <= file->size)
	{
		file->pos = newPos;
		res = 0;
	}

	return res;
}


int64_t NFilePos(NFile *file)
{
#if defined(NEUTON_USE_STDIO)
	if (file->desc)
		return ftell(file->desc);
#endif

	return file->size ? (int64_t)file->pos : -1;
}


uint32_t NFileRead(void *data, uint32_t size, uint32_t count, NFile *file)
{
#if defined(NEUTON_USE_STDIO)
	if (file->desc)
		return fread(data, size, count, file->desc);
#endif

	uint32_t res = count;
	if (file->pos + size * count > file->size)
		res = (file->size - file->pos) / size;

	if (res > 0)
	{
		count = res * size;
		memcpy(data, file->data + file->pos, res * size);
		file->pos += count;
	}

	return res;
}


/**
 * \brief Get data associated with file
 * \param file - file descriptor
 * \return pointer to data or 0 if file not associated with buffer
 */
static inline void* NFileData(NFile* file)
{
	return (file && file->data) ? file->data : NULL;
}


/**
 * \brief Get file size
 * \param file - file descriptor
 * \return file size
 */
static inline uint32_t NFileSize(NFile* file)
{
	return file->size;
}


static uint32_t crc32c(uint32_t crc, const uint8_t* buffer, size_t size)
{
	static const uint32_t POLY = 0xedb88320;

	crc = ~crc;

	while (size--)
	{
		crc ^= *buffer++;
		for (uint32_t k = 0; k < 8; k++)
			crc = crc & 1 ? (crc >> 1) ^ POLY : crc >> 1;
	}

	return ~crc;
}


static void Reverse2BytesValuesBuffer(void* buf, uint32_t valuesCount)
{
	uint16_t* n = buf;

	for (uint32_t i = 0; i < valuesCount; ++i)
	{
		*n = (*n & 0x00FF) << 8 | (*n & 0xFF00) >> 8;
		++n;
	}
}


static void Reverse4BytesValuesBuffer(void* buf, uint32_t valuesCount)
{
	uint32_t* n = buf;

	for (uint32_t i = 0; i < valuesCount; ++i)
	{
		*n = (*n & 0x000000FFu) << 24 | (*n & 0x0000FF00u) << 8 |
			 (*n & 0x00FF0000u) >> 8 | (*n & 0xFF000000u) >> 24;
		++n;
	}
}


/**
 * \brief Check file header on correct format
 * \param file - descriptor of the file
 * \param reverseByteOrder - output parameter; flag of the need to reverse
 *        byte order for correct data reading
 */
static Err CheckFileHeader(NFile *file, uint8_t* reverseByteOrder, uint8_t type)
{
	BinHeader header;
	const uint32_t oneElement = 1;

	if (NFileSeek(file, 0, SEEK_SET) != 0)
		return ERR_READ_FILE;

	if (NFileRead(&header, sizeof(header), oneElement, file) != oneElement)
		return ERR_READ_FILE;

	if (!(header.nb[0] == 'n' && header.nb[1] == 'b'))
		return ERR_BAD_FILE_FORMAT;

	if (header.type != type)
		return ERR_BAD_FILE_FORMAT;

	const uint16_t BOM_PATTERN = 0xABCD;

	if (header.bom == BOM_PATTERN)
	{
		*reverseByteOrder = 0;
	}
	else
	{
		if (((header.bom & 0x00FF) << 8 | (header.bom & 0xFF00) >> 8) == BOM_PATTERN)
		{
			*reverseByteOrder = 1;
		}
		else
		{
			return ERR_BAD_FILE_FORMAT;
		}
	}

	Err err = ERR_NO_ERROR;

	if (type == TYPE_MODEL)
	{
		if (NFileSeek(file, 0, SEEK_SET) != 0)
			return ERR_READ_FILE;

		uint32_t crcActual = 0;
		for (uint32_t pos = 0; pos < NFileSize(file) - sizeof(crcActual); ++pos)
		{
			uint8_t byte;
			if (NFileRead(&byte, sizeof(byte), oneElement, file) != oneElement)
				return ERR_READ_FILE;

			crcActual = crc32c(crcActual, &byte, sizeof(byte));
		}

		uint32_t crcFromFile = 0;
		if (NFileRead(&crcFromFile, sizeof(crcFromFile), oneElement, file) != oneElement)
			return ERR_BAD_FILE_FORMAT;

		if (*reverseByteOrder)
			Reverse4BytesValuesBuffer(&crcFromFile, oneElement);

		if (NFileSeek(file, sizeof(header), SEEK_SET) != 0)
			return ERR_READ_FILE;

		err = (crcActual == crcFromFile) ? ERR_NO_ERROR : ERR_INCONSISTENT_DATA;
	}

	return err;
}

#if 0
static void Reverse8BytesValuesBuffer(void* buf, uint32_t valuesCount)
{
	uint64_t* n = buf;

	for (uint32_t i = 0; i < valuesCount; ++i)
	{
		Reverse4BytesValuesBuffer(n, 2);
		*n = (*n & 0x00000000FFFFFFFFull) << 32 | (*n & 0xFFFFFFFF00000000ull) >> 32;
		++n;
	}
}
#endif


static inline uint32_t AlignBy(uint8_t align, size_t value)
{
	return (value % align == 0) ? 0 : align - (value % align);
}


Err NLoadModel(NFile *file, NeuralNet *model, uint8_t copy)
{
	Err err = ERR_NO_ERROR;

	if (!file)
		return ERR_OPEN_FILE;
	if (!model)
		return ERR_BAD_ARGUMENT;


	void* data = model->data;
	NFreeModel(model);
	model->data = data;


	if (CheckFileHeader(file, &model->reverseByteOrder, TYPE_MODEL) != ERR_NO_ERROR)
		return ERR_BAD_FILE_FORMAT;


	const uint8_t oneElement = 1;


	MetaInfo metaInfo = { 0 };
	if (NFileRead(&metaInfo, sizeof(metaInfo), oneElement, file) != oneElement)
		return ERR_READ_FILE;

	uint32_t weightsDim = 0;
	if (NFileRead(&weightsDim, sizeof(weightsDim), oneElement, file) != oneElement)
		return ERR_READ_FILE;

	if (!(metaInfo.quantisation == 8
#if (NEUTON_Q16_SUPPORT == 1)
		  || metaInfo.quantisation == 16
#endif
#if (NEUTON_Q32_SUPPORT == 1)
		  || metaInfo.quantisation == 32
#endif
	))
		return ERR_FEATURE_NOT_SUPPORTED;

	if (!weightsDim || !metaInfo.inputsDim || !metaInfo.outputsDim || !metaInfo.neuronsCount)
		return ERR_INCONSISTENT_DATA;

	if (model->reverseByteOrder)
	{
		Reverse2BytesValuesBuffer(&metaInfo.inputsDim,    oneElement);
		Reverse2BytesValuesBuffer(&metaInfo.outputsDim,   oneElement);
		Reverse2BytesValuesBuffer(&metaInfo.neuronsCount, oneElement);
		Reverse4BytesValuesBuffer(&weightsDim,            oneElement);
	}

	model->options              = metaInfo.options;
	model->taskType             = metaInfo.taskType;
	model->inputsDim            = metaInfo.inputsDim;
	model->outputsDim           = metaInfo.outputsDim;
	model->quantisation         = metaInfo.quantisation;

	model->neuronsCount         = metaInfo.neuronsCount;
	model->weightDim            = weightsDim;

	const uint8_t align            = model->quantisation / 8;
	const uint8_t positionTypeSize = sizeof(*model->links);
	const uint8_t offsetTypeSize   = model->weightDim <= 256 ? 1 : model->weightDim <= 65536 ? 2 : 4;
	const uint8_t limitTypeSize    = sizeof(*model->inputsMin);
	const uint8_t coeffTypeSize    = (model->quantisation == 32) ? 4 : model->quantisation == 16 ? 2 : 1;
	const uint8_t accTypeSize      = coeffTypeSize;

	if (!positionTypeSize || !coeffTypeSize || !limitTypeSize || !pointerTypeSize || !align)
		return ERR_MEMORY_ALLOCATION;

	uint16_t inputLimitsCount =
			(model->options & BIT_ONE_MAXMIN_FOR_ALL_INPUTS) > 0 ? oneElement : model->inputsDim;

	uint8_t hasLogScale = (model->options & BIT_LOG_SCALE_OUT_EXISTS) > 0;


	// This part can be mapped
	uint32_t blockSize =
		2           * limitTypeSize * inputLimitsCount +  // input limits
		2           * limitTypeSize * model->outputsDim + // output limits
		hasLogScale * limitTypeSize * model->outputsDim;  // output log scale

	blockSize +=
		AlignBy(align, blockSize) +
		model->outputsDim * positionTypeSize;             // output neuron indexes
	blockSize +=
		AlignBy(align, blockSize) +
		2 * model->neuronsCount * positionTypeSize;       // int/ext links count
	blockSize +=
		AlignBy(align, blockSize) +
		model->weightDim * positionTypeSize;              // model links
	blockSize +=
		AlignBy(align, blockSize) +
		model->weightDim * coeffTypeSize;                 // model weights
	blockSize +=
		AlignBy(align, blockSize) +
		model->neuronsCount * coeffTypeSize;              // activation function coefficients


	uint8_t useMapper = !copy && (model->reverseByteOrder == 0) && NFileData(file) &&
						(NFileSize(file) >= (blockSize + NFilePos(file)));
	if (useMapper)
		blockSize = 0;

	// This part always in RAM
	const uint8_t memAlign = pointerTypeSize;

	blockSize +=
		AlignBy(memAlign, blockSize) +
		model->outputsDim * limitTypeSize;                // output buffer

	blockSize +=
		AlignBy(memAlign, blockSize) +
		model->neuronsCount * accTypeSize;                // accumulators

	blockSize +=
		AlignBy(memAlign, blockSize) +
		2 * model->neuronsCount * offsetTypeSize;         // int/ext model links


	uint8_t* block = model->memoryBlock = NAlloc(oneElement, blockSize);
	if (block == NULL)
		return ERR_MEMORY_ALLOCATION;


	// This part can be mapped
	if (useMapper)
		block = (uint8_t*) NFileData(file) + NFilePos(file);

	model->inputsMax  = (void*) block; block += limitTypeSize * inputLimitsCount;
	model->inputsMin  = (void*) block; block += limitTypeSize * inputLimitsCount;
	model->outputsMax = (void*) block; block += limitTypeSize * model->outputsDim;
	model->outputsMin = (void*) block; block += limitTypeSize * model->outputsDim;

	model->outputsLogOffset = hasLogScale ? (void*) block : NULL;
	block += hasLogScale * limitTypeSize * model->outputsDim;

	block += AlignBy(align, (size_t) block);
	model->outputLabels = (void*) block; block += positionTypeSize * model->outputsDim;

	block += AlignBy(align, (size_t) block);
	model->intLinksCounters = (void*) block; block += positionTypeSize * model->neuronsCount;
	model->extLinksCounters = (void*) block; block += positionTypeSize * model->neuronsCount;

	uint32_t structureSize = positionTypeSize * model->weightDim;
	uint32_t structureOffset = structureSize += AlignBy(align, structureSize);
	structureSize += coeffTypeSize * model->weightDim;

	block += AlignBy(align, (size_t) block);
	void* structure = model->links = (void*) block; block += structureSize;
	model->weights.raw = (void*) ((uint8_t*) structure + structureOffset);

	block += AlignBy(align, (size_t) block);
	model->fncCoeffs.raw = block; block += coeffTypeSize * model->neuronsCount;


	// This part always in RAM
	if (useMapper)
		block = model->memoryBlock;

	block += AlignBy(memAlign, (size_t) block);
	model->outputBuffer = (void*) block; block += limitTypeSize * model->outputsDim;

	block += AlignBy(memAlign, (size_t) block);
	model->accumulators.raw = (void*) block; block += accTypeSize * model->neuronsCount;

	block += AlignBy(memAlign, (size_t) block);
	model->intLinks.u8 = block; block += offsetTypeSize * model->neuronsCount;
	model->extLinks.u8 = block; block += offsetTypeSize * model->neuronsCount;

	if (!useMapper)
	{
		if (NFileRead(model->inputsMax,  limitTypeSize, inputLimitsCount, file) != inputLimitsCount ||
			NFileRead(model->inputsMin,  limitTypeSize, inputLimitsCount, file) != inputLimitsCount ||
			NFileRead(model->outputsMax, limitTypeSize, model->outputsDim, file) != model->outputsDim ||
			NFileRead(model->outputsMin, limitTypeSize, model->outputsDim, file) != model->outputsDim)
			return ERR_READ_FILE;

		if (hasLogScale &&
			NFileRead(model->outputsLogOffset, limitTypeSize, model->outputsDim, file) != model->outputsDim)
			return ERR_READ_FILE;

		if (NFileSeek(file, AlignBy(align, NFilePos(file)), SEEK_CUR) == 0 &&
			NFileRead(model->outputLabels, positionTypeSize, model->outputsDim, file) != model->outputsDim)
			return ERR_READ_FILE;

		if (NFileSeek(file, AlignBy(align, NFilePos(file)), SEEK_CUR) == 0 &&
			(NFileRead(model->intLinksCounters, positionTypeSize, model->neuronsCount, file) != model->neuronsCount ||
			NFileRead(model->extLinksCounters, positionTypeSize, model->neuronsCount, file) != model->neuronsCount))
			return ERR_READ_FILE;

		if (NFileSeek(file, AlignBy(align, NFilePos(file)), SEEK_CUR) == 0 &&
			NFileRead(structure, structureSize, oneElement, file) != oneElement)
			return ERR_READ_FILE;

		if (NFileSeek(file, AlignBy(align, NFilePos(file)), SEEK_CUR) == 0 &&
			NFileRead(model->fncCoeffs.raw,   coeffTypeSize, model->neuronsCount, file) != model->neuronsCount)
			return ERR_READ_FILE;

		if (model->reverseByteOrder)
		{
			switch (limitTypeSize)
			{
			case 4:
				Reverse4BytesValuesBuffer(model->inputsMax,  inputLimitsCount);
				Reverse4BytesValuesBuffer(model->inputsMin,  inputLimitsCount);
				Reverse4BytesValuesBuffer(model->outputsMax, model->outputsDim);
				Reverse4BytesValuesBuffer(model->outputsMin, model->outputsDim);
				if (hasLogScale)
					Reverse4BytesValuesBuffer(model->outputsLogOffset, model->outputsDim);
				break;

			default:
				return ERR_FEATURE_NOT_SUPPORTED;
				break;
			}

			switch (positionTypeSize)
			{
			case 1:
				break;

			case 2:
				Reverse2BytesValuesBuffer(model->intLinksCounters, model->neuronsCount);
				Reverse2BytesValuesBuffer(model->extLinksCounters, model->neuronsCount);
				Reverse2BytesValuesBuffer(structure,               model->weightDim);
				Reverse2BytesValuesBuffer(model->outputLabels,     model->outputsDim);
				break;

			case 4:
				Reverse4BytesValuesBuffer(model->intLinksCounters, model->neuronsCount);
				Reverse4BytesValuesBuffer(model->extLinksCounters, model->neuronsCount);
				Reverse4BytesValuesBuffer(structure,               model->weightDim);
				Reverse4BytesValuesBuffer(model->outputLabels,     model->outputsDim);
				break;

			default:
				return ERR_FEATURE_NOT_SUPPORTED;
				break;
			}

			switch (coeffTypeSize)
			{
			case 1:
				break;

			case 2:
				Reverse2BytesValuesBuffer((uint8_t*) structure + structureOffset, model->weightDim);
				Reverse2BytesValuesBuffer(model->fncCoeffs.raw,                   model->neuronsCount);
				break;

			case 4:
				Reverse4BytesValuesBuffer((uint8_t*) structure + structureOffset, model->weightDim);
				Reverse4BytesValuesBuffer(model->fncCoeffs.raw,                   model->neuronsCount);
				break;

			default:
				return ERR_FEATURE_NOT_SUPPORTED;
				break;
			}

		}
	}


	uint32_t offset = 0;
	for (uint32_t idx = 0; idx < model->neuronsCount; offset += model->intLinksCounters[idx++])
	{
		switch (offsetTypeSize)
		{
		case 4:	 model->intLinks.u32[idx] = offset; break;
		case 2:	 model->intLinks.u16[idx] = offset; break;
		case 1:	 model->intLinks.u8[idx]  = offset; break;
		default: return ERR_FEATURE_NOT_SUPPORTED;
		}
	}

	for (uint32_t idx = 0; idx < model->neuronsCount; offset += model->extLinksCounters[idx++])
	{
		switch (offsetTypeSize)
		{
		case 4:	 model->extLinks.u32[idx] = offset; break;
		case 2:	 model->extLinks.u16[idx] = offset; break;
		case 1:	 model->extLinks.u8[idx]  = offset; break;
		default: return ERR_FEATURE_NOT_SUPPORTED;
		}
	}


	for (uint32_t idx = 0; idx < model->outputsDim; idx++)
	{
		if (model->outputLabels[idx] >= model->neuronsCount)
			return ERR_INCONSISTENT_DATA;

		if (model->outputsMin[idx] > model->outputsMax[idx])
			return ERR_INCONSISTENT_DATA;
	}

	for (uint32_t idx = 0; idx < inputLimitsCount; idx++)
	{
		if (model->inputsMin[idx] > model->inputsMax[idx])
			return ERR_INCONSISTENT_DATA;
	}

	if ((inputLimitsCount == oneElement) && (model->inputsMax[0] != model->inputsMin[0]))
		model->cachedInputsDiff = model->inputsMax[0] - model->inputsMin[0];


	NFileClose(file);


	return err;
}


Err NLoadModelEx(const char* fileName, NeuralNet* model)
{
	return NLoadModel(NFileOpen(fileName, "rb"), model, 1);
}


void NFreeModel(NeuralNet* model)
{
	if (model)
	{
		if (model->memoryBlock)
			NFree(model->memoryBlock);

		memset(model, 0, sizeof(*model));
	}
}


void NNormalizeSample(float* sample, NeuralNet* model)
{
	uint16_t inputLimitsCount =
			(model->options & BIT_ONE_MAXMIN_FOR_ALL_INPUTS) > 0
			? 1 : model->inputsDim;

	for (uint16_t i = 0; i < model->inputsDim - 1; ++i)
	{
		if (inputLimitsCount == 1)
		{
			if (model->cachedInputsDiff)
			{
				sample[i] = (sample[i] - model->inputsMin[0]) / model->cachedInputsDiff;
			}
			else if (model->inputsMax[0] != model->inputsMin[0])
			{
				sample[i] = (sample[i] - model->inputsMin[0]) /
						(model->inputsMax[0] - model->inputsMin[0]);
			}

		}
		else if (model->inputsMax[i] != model->inputsMin[i])
		{
			sample[i] = (sample[i] - model->inputsMin[i]) /
					(model->inputsMax[i] - model->inputsMin[i]);
		}

		if (sample[i] > 1.0f)
			sample[i] = 1.0f;

		if (sample[i] < 0.0f)
			sample[i] = 0.0f;
	}
}


void NDenormalizeResult(float* result, NeuralNet* model)
{
	if (model->taskType == TASK_BINARY_CLASSIFICATION)
	{
		float sum = 0;

		for (uint16_t i = 0; i < model->outputsDim; ++i)
			sum += (float) result[i];

		for (uint16_t i = 0; i < model->outputsDim; ++i)
			result[i] = (float) result[i] / sum;
	}

	if (model->taskType == TASK_MULTICLASS_CLASSIFICATION || model->taskType == TASK_REGRESSION)
	{
		uint8_t hasLogScale = (model->options & BIT_LOG_SCALE_OUT_EXISTS) > 0;

		for (uint16_t i = 0; i < model->outputsDim; ++i)
		{
			result[i] = result[i] * (model->outputsMax[i] - model->outputsMin[i]) +
					model->outputsMin[i];

			if (hasLogScale && (model->outputsLogOffset[i] != 0xFFFFFFFF))
			{
				result[i] = exp(result[i]) - model->outputsLogOffset[i];
			}
		}
	}
}


static inline uint32_t valueAt(uint32_t index, Pointer p, uint8_t typeSize)
{
	switch (typeSize)
	{
	case sizeof(uint8_t):  return p.u8[index];
	case sizeof(uint16_t): return p.u16[index];
	case sizeof(uint32_t): return p.u32[index];
	default:               return 0;
	}
}


static uint8_t accurate_fast_sigmoid_u8(int32_t arg)
{
	uint8_t qResult = 0;
	uint8_t secondPointY = 0;
	uint8_t firstPointY = 0;

	const uint8_t QLVL = 8;
	const int32_t CT_MAX_VALUE = 1u << QLVL;
	const int32_t intPart = abs(arg) / (2u << (QLVL - 1));
	const int32_t realPart = abs(arg) - (intPart << QLVL);

	if (intPart == 0 && realPart == 0)
	{
		return ldexp(0.5, QLVL);
	}

	int s = arg > 0 ? 0 : 1;
	if (realPart == 0)
	{
		for (int i = 0; i < QLVL; i++)
		{
			const uint8_t bit = ((i / intPart + s) % 2);
			qResult = qResult | (bit << (QLVL - i - 1));
		}
		return qResult;
	}

	const int32_t secondPointX = intPart + 1;
	if (intPart == 0)
	{
		firstPointY = ldexp(0.5, QLVL);
		for (int i = 0; i < QLVL; i++)
		{
			const uint8_t bit = ((i / secondPointX) % 2);
			secondPointY = secondPointY | (bit << (QLVL - i - 1));
		}
	}
	else
	{
		if (secondPointX == 0)
		{
			for (int i = 0; i < QLVL; i++)
			{
				const uint8_t bit = ((i / intPart) % 2);
				firstPointY = firstPointY | (bit << (QLVL - i - 1));
			}
			secondPointY = ldexp(0.5, QLVL);
		}
		else
		{
			for (int i = 0; i < QLVL; i++)
			{
				uint8_t bit = ((i / intPart) % 2);
				firstPointY = firstPointY | (bit << (QLVL - i - 1));
				bit = ((i / secondPointX) % 2);
				secondPointY = secondPointY | (bit << (QLVL - i - 1));
			}
		}
	}

	const int32_t res = (CT_MAX_VALUE - realPart) * firstPointY + realPart * secondPointY;
	if (arg > 0)
	{
		return res >> QLVL;
	}
	else
	{
		qResult = res >> QLVL;
		return qResult == 0 ? CT_MAX_VALUE - 1 : CT_MAX_VALUE - qResult;
	}
}

#if (NEUTON_Q16_SUPPORT == 1)
static uint16_t accurate_fast_sigmoid_u16(int64_t arg)
{
	uint16_t qResult = 0;
	uint16_t secondPointY = 0;
	uint16_t firstPointY = 0;

	const uint8_t QLVL = 16;
	const int64_t CT_MAX_VALUE = 1ul << QLVL;
	const int64_t intPart = labs(arg) / (2ul << (QLVL - 1));
	const int64_t realPart = labs(arg) - (intPart << QLVL);

	if (intPart == 0 && realPart == 0)
	{
		return ldexp(0.5, QLVL);
	}

	int s = arg > 0 ? 0 : 1;
	if (realPart == 0)
	{
		for (int i = 0; i < QLVL; i++)
		{
			const uint8_t bit = ((i / intPart + s) % 2);
			qResult = qResult | (bit << (QLVL - i - 1));
		}
		return qResult;
	}

	const int64_t secondPointX = intPart + 1;
	if (intPart == 0)
	{
		firstPointY = ldexp(0.5, QLVL);
		for (int i = 0; i < QLVL; i++)
		{
			const uint8_t bit = ((i / secondPointX) % 2);
			secondPointY = secondPointY | (bit << (QLVL - i - 1));
		}
	}
	else
	{
		if (secondPointX == 0)
		{
			for (int i = 0; i < QLVL; i++)
			{
				const uint8_t bit = ((i / intPart) % 2);
				firstPointY = firstPointY | (bit << (QLVL - i - 1));
			}
			secondPointY = ldexp(0.5, QLVL);
		}
		else
		{
			for (int i = 0; i < QLVL; i++)
			{
				uint8_t bit = ((i / intPart) % 2);
				firstPointY = firstPointY | (bit << (QLVL - i - 1));
				bit = ((i / secondPointX) % 2);
				secondPointY = secondPointY | (bit << (QLVL - i - 1));
			}
		}
	}

	const int64_t res = (CT_MAX_VALUE - realPart) * firstPointY + realPart * secondPointY;
	if (arg > 0)
	{
		return res >> QLVL;
	}
	else
	{
		qResult = res >> QLVL;
		return qResult == 0 ? CT_MAX_VALUE - 1 : CT_MAX_VALUE - qResult;
	}
}
#endif // NEUTON_Q16_SUPPORT


static inline float dequantiseValue(int32_t value, NeuralNet* model)
{
	return (float) value / (float) (2 << (model->quantisation - 1));
}


static inline float* RunInferenceQ8(NeuralNet* model, float* inputs)
{
	const uint8_t offsetTypeSize = model->weightDim <= 256 ? 1 : model->weightDim <= 65536 ? 2 : 4;
	uint32_t offset;

	memset(model->accumulators.raw, 0, model->neuronsCount * sizeof(*model->accumulators.u8));

	for (uint32_t neuronIndex = 0; neuronIndex < model->neuronsCount; neuronIndex++)
	{
		int32_t summ = 0;

		offset = valueAt(neuronIndex, model->intLinks, offsetTypeSize);
		for (uint16_t idx = 0; idx < model->intLinksCounters[neuronIndex]; ++idx)
		{
			const int32_t firstValue  = (int32_t) model->weights.i8[offset+idx];
			const int32_t secondValue = (int32_t) model->accumulators.u8[model->links[offset+idx]];
			summ += firstValue * secondValue;
		}

		offset = valueAt(neuronIndex, model->extLinks, offsetTypeSize);
		for (uint16_t idx = 0; idx < model->extLinksCounters[neuronIndex]; ++idx)
		{
			const int32_t firstValue  = (int32_t) model->weights.i8[offset+idx];
			const int32_t secondValue = (int32_t) ldexp(inputs[model->links[offset+idx]] > MAX_INPUT_FLOAT
					? MAX_INPUT_FLOAT : inputs[model->links[offset+idx]], 8);
			summ += firstValue * secondValue;
		}

		if (model->options & BIT_FORCE_INTEGER_CALCULATIONS)
		{
			model->accumulators.u8[neuronIndex] = accurate_fast_sigmoid_u8(
				-(((int32_t) model->fncCoeffs.u8[neuronIndex] * summ) >> (8 + KSHIFT_2 - 1))
			);
		}
		else
		{
			const float qs = (float) (((int32_t) model->fncCoeffs.u8[neuronIndex] * summ)
					>> (8 + KSHIFT_2 - 1)) / (float) (2u << 7);
			const float tmpValue = 1.0f / (1.0f + expf(-qs));
			model->accumulators.u8[neuronIndex] = ldexp(tmpValue > MAX_INPUT_FLOAT ? MAX_INPUT_FLOAT : tmpValue, 8);
		}

	}

	for (uint16_t idx = 0; idx < model->outputsDim; idx++)
		model->outputBuffer[idx] = dequantiseValue(model->accumulators.u8[model->outputLabels[idx]], model);

	return model->outputBuffer;
}


#if (NEUTON_Q16_SUPPORT == 1)
static inline float* RunInferenceQ16(NeuralNet* model, float* inputs)
{
	const uint8_t offsetTypeSize = model->weightDim <= 256 ? 1 : model->weightDim <= 65536 ? 2 : 4;
	uint32_t offset;

	memset(model->accumulators.raw, 0, model->neuronsCount * sizeof(*model->accumulators.u16));

	for (uint32_t neuronIndex = 0; neuronIndex < model->neuronsCount; neuronIndex++)
	{
		int64_t summ = 0;

		offset = valueAt(neuronIndex, model->intLinks, offsetTypeSize);
		for (uint16_t idx = 0; idx < model->intLinksCounters[neuronIndex]; ++idx)
		{
			const int64_t firstValue  = (int64_t) model->weights.i16[offset+idx];
			const int64_t secondValue = (int64_t) model->accumulators.u16[model->links[offset+idx]];
			summ += firstValue * secondValue;
		}

		offset = valueAt(neuronIndex, model->extLinks, offsetTypeSize);
		for (uint16_t idx = 0; idx < model->extLinksCounters[neuronIndex]; ++idx)
		{
			const int64_t firstValue  = (int64_t) model->weights.i16[offset+idx];
			const int64_t secondValue = (int64_t) ldexp(inputs[model->links[offset+idx]] > MAX_INPUT_FLOAT
					? MAX_INPUT_FLOAT : inputs[model->links[offset+idx]], 16);
			summ += firstValue * secondValue;
		}

		if (model->options & BIT_FORCE_INTEGER_CALCULATIONS)
		{
			model->accumulators.u16[neuronIndex] = accurate_fast_sigmoid_u16(
				-(((int64_t) model->fncCoeffs.u16[neuronIndex] * summ) >> (16 + KSHIFT_10 - 1))
			);
		}
		else
		{
			const float qs = (float) (((int64_t) model->fncCoeffs.u16[neuronIndex] * summ)
					>> (16 + KSHIFT_10 - 1)) / (float) (2u << 15);
			const float tmpValue = 1.0f / (1.0f + expf(-qs));
			model->accumulators.u16[neuronIndex] = ldexp(tmpValue > MAX_INPUT_FLOAT ? MAX_INPUT_FLOAT : tmpValue, 16);
		}
	}

	for (uint16_t idx = 0; idx < model->outputsDim; idx++)
		model->outputBuffer[idx] = dequantiseValue(model->accumulators.u16[model->outputLabels[idx]], model);

	return model->outputBuffer;
}
#endif


#if (NEUTON_Q32_SUPPORT == 1)
static inline float* RunInferenceF32(NeuralNet* model, float* inputs)
{
	const uint8_t offsetTypeSize = model->weightDim <= 256 ? 1 : model->weightDim <= 65536 ? 2 : 4;
	uint32_t offset;

	memset(model->accumulators.raw, 0, model->neuronsCount * sizeof(*model->accumulators.f32));

	for (uint32_t neuronIndex = 0; neuronIndex < model->neuronsCount; neuronIndex++)
	{
		double summ = 0;

		offset = valueAt(neuronIndex, model->intLinks, offsetTypeSize);
		for (uint16_t idx = 0; idx < model->intLinksCounters[neuronIndex]; ++idx)
		{
			const double firstValue  = (double) model->weights.f32[offset+idx];
			const double secondValue = (double) model->accumulators.f32[model->links[offset+idx]];
			summ += firstValue * secondValue;
		}

		offset = valueAt(neuronIndex, model->extLinks, offsetTypeSize);
		for (uint16_t idx = 0; idx < model->extLinksCounters[neuronIndex]; ++idx)
		{
			const double firstValue  = (double) model->weights.f32[offset+idx];
			const double secondValue = (double) inputs[model->links[offset+idx]];
			summ += firstValue * secondValue;
		}

		model->accumulators.f32[neuronIndex] =
				1.0f / (1.0f + exp((double) ((double) -model->fncCoeffs.f32[neuronIndex]) * summ));
	}

	for (uint16_t idx = 0; idx < model->outputsDim; idx++)
		model->outputBuffer[idx] = model->accumulators.f32[model->outputLabels[idx]];

	return model->outputBuffer;
}
#endif


float* NRunInference(NeuralNet* model, float* inputs)
{
	switch (model->quantisation)
	{
	case 8:  return RunInferenceQ8 (model, inputs);

#if (NEUTON_Q16_SUPPORT == 1)
	case 16: return RunInferenceQ16(model, inputs);
#endif

#if (NEUTON_Q32_SUPPORT == 1)
	case 32: return RunInferenceF32(model, inputs);
#endif

	default: return NULL;
	}
}


Err NOpenDataset(NFile *file, Dataset *dataset)
{
	if (!file || !dataset)
		return ERR_BAD_ARGUMENT;

	dataset->file = file;
	dataset->reverseByteOrder = 0;

	if (CheckFileHeader(dataset->file, &dataset->reverseByteOrder, TYPE_DATASET) != ERR_NO_ERROR)
		return ERR_BAD_FILE_FORMAT;

	const uint32_t oneElement = 1;
	const uint32_t headerSize = sizeof(BinHeader);
	const uint64_t metadataAddressSize;

	if (NFileSeek(dataset->file, headerSize, SEEK_SET) != 0)
		return ERR_BAD_FILE_FORMAT;

	if (NFileRead(&dataset->endDataPos, sizeof(dataset->endDataPos), oneElement, dataset->file) != oneElement)
		return ERR_READ_FILE;
	if (dataset->reverseByteOrder)
		Reverse4BytesValuesBuffer(&dataset->endDataPos, oneElement);

	if (NFileSeek(dataset->file, dataset->endDataPos, SEEK_SET) != 0)
		return ERR_BAD_FILE_FORMAT;

	if (NFileRead(&dataset->sampleDim, sizeof(dataset->sampleDim), oneElement, dataset->file) != oneElement)
		return ERR_READ_FILE;
	if (dataset->reverseByteOrder)
		Reverse4BytesValuesBuffer(&dataset->sampleDim, oneElement);

	if (NFileSeek(dataset->file, headerSize + sizeof(metadataAddressSize), SEEK_SET) != 0)
		return ERR_BAD_FILE_FORMAT;

	return ERR_NO_ERROR;
}


Err NOpenDatasetEx(const char* datasetFilename, Dataset* dataset)
{
	NFile* file = NFileOpen(datasetFilename, "rb");
	return NOpenDataset(file, dataset);
}


void NCloseDataset(Dataset* dataset)
{
	if (dataset)
	{
		if (dataset->file)
			NFileClose(dataset->file);

		memset(dataset, 0, sizeof(*dataset));
	}
}


Err NReadDatasetSample(Dataset* dataset, float* sample, uint32_t* readSamples)
{
	*readSamples = 0;

	int64_t currentPos = NFilePos(dataset->file);
	if (currentPos == -1)
		return ERR_READ_FILE;

	if (currentPos < dataset->endDataPos)
	{
		if (NFileRead(sample, sizeof(*sample), dataset->sampleDim, dataset->file) != dataset->sampleDim)
			return ERR_READ_FILE;

		if (dataset->reverseByteOrder)
			Reverse4BytesValuesBuffer(sample, dataset->sampleDim);

		sample[dataset->sampleDim] = 1.0f;  // set the BIAS value, just in case
		*readSamples = 1;
	}

	return ERR_NO_ERROR;
}


#if defined(NEUTON_MEMORY_BENCHMARK)
static inline uint32_t NAllocCost()
{
	return 8;
}


static void insertChunk(void* ptr, uint32_t size)
{
	for (uint32_t i = 0; i < NEUTON_MEMORY_BENCHMARK_SIZE; ++i)
		if (chunks[i].ptr == NULL)
		{
			size += NAllocCost();
			chunks[i].ptr = ptr;
			chunks[i].size = size;
			memAllocated += size;

			if (++allocs > allocsMax)
				allocsMax = allocs;
			if (memAllocated > memAllocatedMax)
				memAllocatedMax = memAllocated;

			return;
		}

	assert(0);
}


static void removeChunk(void* ptr)
{
	for (uint32_t i = 0; i < NEUTON_MEMORY_BENCHMARK_SIZE; ++i)
		if (chunks[i].ptr == ptr)
		{
			memAllocated -= chunks[i].size;
			memset(&chunks[i], 0, sizeof(MemoryChunk));
			allocs--;
			return;
		}

	assert(0);
}
#endif


void* NAlloc(uint32_t count, uint32_t size)
{
	if (count * size == 0)
		return NULL;

	void* ptr = calloc(count, size);

#if defined(NEUTON_MEMORY_BENCHMARK)
	static uint8_t initialised = 0;
	if (!initialised)
	{
		initialised = 1;
		memAllocated = memAllocatedMax = 0 + _NeutonExtraMemoryUsage();
	}

	if (ptr)
		insertChunk(ptr, count * size);
#endif

	return ptr;
}


void NFree(void* ptr)
{
	free(ptr);

#if defined(NEUTON_MEMORY_BENCHMARK)
	removeChunk(ptr);
#endif
}


uint32_t NBytesAllocated()
{
#if defined(NEUTON_MEMORY_BENCHMARK)
	return memAllocated;
#else
	return 0;
#endif
}


uint32_t NBytesAllocatedTotal()
{
#if defined(NEUTON_MEMORY_BENCHMARK)
	return memAllocatedMax;
#else
	return 0;
#endif
}

