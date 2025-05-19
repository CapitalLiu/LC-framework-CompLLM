#ifndef H_HANS_COMMON_H
#define H_HANS_COMMON_H

#include <string.h>
#include <assert.h>

// 数据类型定义
#define DATA_TYPE_8BIT 1
#define DATA_TYPE_16BIT 2

// 常量定义
#define MAX_SYMBOLS_8BIT 256
#define MAX_SYMBOLS_16BIT 65536
#define BITLEN_N_8BIT 3  // 8位数据的元数据位数
#define BITLEN_N_16BIT 4  // 16位数据的元数据位数

// 位操作辅助函数
static inline void hans_append_bit(byte* data, int& bit_pos, bool bit) {
  int byte_pos = bit_pos / 8;
  int bit_offset = bit_pos % 8;
  
  if (bit) {
    data[byte_pos] |= (1 << (7 - bit_offset));
  } else {
    data[byte_pos] &= ~(1 << (7 - bit_offset));
  }
  
  bit_pos++;
}

static inline void hans_append_bits(byte* data, int& bit_pos, uint32_t value, int num_bits) {
  for (int i = num_bits - 1; i >= 0; i--) {
    hans_append_bit(data, bit_pos, (value >> i) & 1);
  }
}

#endif // H_HANS_COMMON_H