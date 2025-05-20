#include "include/h_HANS.h"

static inline bool h_HANS_1(int& csize, byte* in, byte* out)
{
    return h_HANS<uint8_t>(csize, in, out);
}

static inline void h_iHANS_1(int& csize, byte* in, byte* out)
{
  return;
}