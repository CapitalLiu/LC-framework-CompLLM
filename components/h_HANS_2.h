#include "include/h_HANS.h"

static inline bool h_HANS_2(int& csize, byte* in, byte* out)
{
    return h_HANS<uint16_t>(csize, in, out);
}

static inline void h_iHANS_2(int& csize, byte* in, byte* out)
{
  return;
}