#ifndef TKM_STATUS_H
#define TKM_STATUS_H

#include <stdint.h>

typedef enum {
    TKM_OK = 0,
    TKM_ERR = -1,
} TkmStatus;

#define TKM_INVALID_TOKEN UINT16_MAX

#endif
