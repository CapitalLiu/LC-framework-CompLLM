#ifndef H_HANS_1_H
#define H_HANS_1_H

#include "./include/h_HANS_common.h"

// 8位数据特定设置
#define H1_MAX_SYMBOLS MAX_SYMBOLS_8BIT
#define H1_BITLEN_N BITLEN_N_8BIT
#define H1_ELEMENT_SIZE 1

// 构建频率表 - 8位版本
static inline void h1_build_frequency_table(const byte* data, int size, uint32_t* freq) {
  // 初始化频率计数
  memset(freq, 0, H1_MAX_SYMBOLS * sizeof(uint32_t));
  
  // 计算频率
  for (int i = 0; i < size; i++) {
    freq[data[i]]++;
  }
}

static inline bool h_HANS_1(int& csize, const byte in[CS], byte out[CS]) {
  if (csize < 4) return true;  // 太小的数据不压缩
  
  // 计算元素总数
  int total_elements = csize / H1_ELEMENT_SIZE;
  
  // 使用临时数组存储频率和编码映射
  uint32_t* freq = (uint32_t*)malloc(H1_MAX_SYMBOLS * sizeof(uint32_t));
  uint32_t* symbol_to_code = (uint32_t*)malloc(H1_MAX_SYMBOLS * sizeof(uint32_t));
  uint32_t* code_to_symbol = (uint32_t*)malloc(H1_MAX_SYMBOLS * sizeof(uint32_t));
  
  if (!freq || !symbol_to_code || !code_to_symbol) {
    if (freq) free(freq);
    if (symbol_to_code) free(symbol_to_code);
    if (code_to_symbol) free(code_to_symbol);
    return false;
  }
  
  // 构建频率表
  h1_build_frequency_table(in, csize, freq);
  
  // 创建符号-频率对
  typedef struct {
    uint32_t symbol;
    uint32_t frequency;
  } SymbolFreq;
  
  SymbolFreq* symbol_freqs = (SymbolFreq*)malloc(H1_MAX_SYMBOLS * sizeof(SymbolFreq));
  if (!symbol_freqs) {
    free(freq);
    free(symbol_to_code);
    free(code_to_symbol);
    return false;
  }
  
  int unique_count = 0;
  
  // 收集非零频率的符号
  for (uint32_t i = 0; i < H1_MAX_SYMBOLS; i++) {
    if (freq[i] > 0) {
      symbol_freqs[unique_count].symbol = i;
      symbol_freqs[unique_count].frequency = freq[i];
      unique_count++;
    }
  }
  
  // 按频率排序（降序）
  for (int i = 0; i < unique_count; i++) {
    for (int j = i + 1; j < unique_count; j++) {
      if (symbol_freqs[j].frequency > symbol_freqs[i].frequency) {
        SymbolFreq temp = symbol_freqs[i];
        symbol_freqs[i] = symbol_freqs[j];
        symbol_freqs[j] = temp;
      }
    }
  }
  
  // 构建编码映射
  for (int i = 0; i < unique_count; i++) {
    symbol_to_code[symbol_freqs[i].symbol] = i;
    code_to_symbol[i] = symbol_freqs[i].symbol;
  }
  
  // 清空输出数组
  memset(out, 0, CS);
  
  // 设置块大小
  const int block_size = 16;  // 默认块大小
  
  // 计算块数
  int num_blocks = total_elements / block_size;
  int remaining = total_elements % block_size;
  
  // 写入总元素数（64位），与Python代码一致
  int bit_pos = 0;
  for (int i = 0; i < 64; i++) {
    bool bit = (total_elements >> (63 - i)) & 1;
    hans_append_bit(out, bit_pos, bit);
  }
  
  // 压缩每个块
  for (int block = 0; block < num_blocks; block++) {
    // 找出当前块中最大编码值
    uint32_t max_code = 0;
    
    for (int i = 0; i < block_size; i++) {
      uint32_t code = symbol_to_code[in[block * block_size + i]];
      if (code > max_code) max_code = code;
    }
    
    // 计算编码所需的位数
    int num_bits = 0;
    uint32_t temp = max_code;
    while (temp > 0) {
      temp >>= 1;
      num_bits++;
    }
    if (num_bits == 0) num_bits = 1;  // 至少需要1位
    
    // 写入元数据（位数）
    hans_append_bits(out, bit_pos, num_bits - 1, H1_BITLEN_N);
    
    // 写入压缩后的数据
    for (int i = 0; i < block_size; i++) {
      uint32_t code = symbol_to_code[in[block * block_size + i]];
      hans_append_bits(out, bit_pos, code, num_bits);
    }
    
    // 检查是否超出输出缓冲区
    if ((bit_pos + 7) / 8 >= CS) {
      free(freq);
      free(symbol_to_code);
      free(code_to_symbol);
      free(symbol_freqs);
      return false;
    }
  }
  
  // 处理剩余数据
  if (remaining > 0) {
    // 找出剩余块中最大编码值
    uint32_t max_code = 0;
    
    for (int i = 0; i < remaining; i++) {
      uint32_t code = symbol_to_code[in[num_blocks * block_size + i]];
      if (code > max_code) max_code = code;
    }
    
    // 计算编码所需的位数
    int num_bits = 0;
    uint32_t temp = max_code;
    while (temp > 0) {
      temp >>= 1;
      num_bits++;
    }
    if (num_bits == 0) num_bits = 1;  // 至少需要1位
    
    // 写入元数据（位数）
    hans_append_bits(out, bit_pos, num_bits - 1, H1_BITLEN_N);
    
    // 写入压缩后的数据
    for (int i = 0; i < remaining; i++) {
      uint32_t code = symbol_to_code[in[num_blocks * block_size + i]];
      hans_append_bits(out, bit_pos, code, num_bits);
    }
    
    // 检查是否超出输出缓冲区
    if ((bit_pos + 7) / 8 >= CS) {
      free(freq);
      free(symbol_to_code);
      free(code_to_symbol);
      free(symbol_freqs);
      return false;
    }
  }
  
  // 更新压缩后的大小
  csize = (bit_pos + 7) / 8;  // 向上取整到字节边界
  
  // 释放内存
  free(freq);
  free(symbol_to_code);
  free(code_to_symbol);
  free(symbol_freqs);
  
  // 将符号表保存到外部文件（这部分需要在调用函数中实现）
  // 在Python代码中，符号表保存在单独的.freq文件中
  
  return true;
}

// 解压函数 - 8位版本 (添加这个函数来解决 h_iHANS_1 未定义的错误)
static inline bool h_iHANS_1(int& csize, const byte in[CS], byte out[CS]) {
  // 读取原始大小
  int original_size;
  memcpy(&original_size, in, sizeof(int));
  
  // 读取唯一符号数
  int unique_count = in[4];
  
  // 读取符号表
  uint32_t* code_to_symbol = (uint32_t*)malloc(unique_count * sizeof(uint32_t));
  if (!code_to_symbol) return false;
  
  for (int i = 0; i < unique_count; i++) {
    code_to_symbol[i] = in[5 + i];
  }
  
  // 计算压缩数据的起始位置
  int header_size = 5 + unique_count;
  int bit_pos = header_size * 8;  // 位位置
  
  // 计算元素总数
  int total_elements = original_size / H1_ELEMENT_SIZE;
  
  // 设置块大小
  const int block_size = 16;  // 默认块大小
  
  // 计算块数
  int num_blocks = total_elements / block_size;
  int remaining = total_elements % block_size;
  
  // 解压每个块
  int out_pos = 0;
  
  for (int block = 0; block < num_blocks; block++) {
    // 读取元数据（位数）
    int num_bits = 0;
    for (int i = 0; i < H1_BITLEN_N; i++) {
      int byte_pos = bit_pos / 8;
      int bit_offset = bit_pos % 8;
      bool bit = (in[byte_pos] >> (7 - bit_offset)) & 1;
      num_bits = (num_bits << 1) | bit;
      bit_pos++;
    }
    num_bits += 1;  // 实际位数
    
    // 读取并解压数据
    for (int i = 0; i < block_size; i++) {
      uint32_t code = 0;
      for (int j = 0; j < num_bits; j++) {
        int byte_pos = bit_pos / 8;
        int bit_offset = bit_pos % 8;
        bool bit = (in[byte_pos] >> (7 - bit_offset)) & 1;
        code = (code << 1) | bit;
        bit_pos++;
      }
      
      // 确保代码在有效范围内
      if (code >= (uint32_t)unique_count) {
        free(code_to_symbol);
        return false;
      }
      
      out[out_pos++] = (byte)code_to_symbol[code];
    }
  }
  
  // 处理剩余数据
  if (remaining > 0) {
    // 读取元数据（位数）
    int num_bits = 0;
    for (int i = 0; i < H1_BITLEN_N; i++) {
      int byte_pos = bit_pos / 8;
      int bit_offset = bit_pos % 8;
      bool bit = (in[byte_pos] >> (7 - bit_offset)) & 1;
      num_bits = (num_bits << 1) | bit;
      bit_pos++;
    }
    num_bits += 1;  // 实际位数
    
    // 读取并解压数据
    for (int i = 0; i < remaining; i++) {
      uint32_t code = 0;
      for (int j = 0; j < num_bits; j++) {
        int byte_pos = bit_pos / 8;
        int bit_offset = bit_pos % 8;
        bool bit = (in[byte_pos] >> (7 - bit_offset)) & 1;
        code = (code << 1) | bit;
        bit_pos++;
      }
      
      // 确保代码在有效范围内
      if (code >= (uint32_t)unique_count) {
        free(code_to_symbol);
        return false;
      }
      
      out[out_pos++] = (byte)code_to_symbol[code];
    }
  }
  
  // 释放内存
  free(code_to_symbol);
  
  // 更新解压后的大小
  csize = original_size;
  
  return true;
}

#endif // H_HANS_1_H