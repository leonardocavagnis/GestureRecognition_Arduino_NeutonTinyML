#ifndef USER_APP_H
#define USER_APP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint8_t model_init();
float*  model_run_inference(float* sample, 
							uint32_t size_in, 
							uint32_t* size_out);

#ifdef __cplusplus
}
#endif

#endif  // USER_APP_H
