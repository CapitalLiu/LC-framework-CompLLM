 #ifndef H_HANS_H
 #define H_HANS_H
 
 #include <stdint.h>
 #include <stdlib.h>
 #include <string.h>
 
 #define HANS_MAX_SYMBOLS_8BIT 256
 #define HANS_MAX_SYMBOLS_16BIT 65536
 #define HANS_BITLEN_N_8BIT 3
 #define HANS_BITLEN_N_16BIT 4
 #define HANS_DEFAULT_BLOCK_SIZE 16
 
 // 位操作辅助结构
 typedef struct {
     uint8_t *data;
     size_t capacity;
     size_t bit_length;
 } HANSBitArray;
 
 // 初始化位数组
 static inline HANSBitArray* hans_bitarray_create(size_t initial_capacity) {
     HANSBitArray *ba = (HANSBitArray*)malloc(sizeof(HANSBitArray));
     if (!ba) return NULL;
     
     ba->capacity = initial_capacity;
     size_t bytes_needed = (initial_capacity + 7) / 8;
     ba->data = (uint8_t*)calloc(bytes_needed, sizeof(uint8_t));
     if (!ba->data) {
         free(ba);
         return NULL;
     }
     ba->bit_length = 0;
     return ba;
 }
 
 // 扩展位数组容量
 static inline bool hans_bitarray_ensure_capacity(HANSBitArray *ba, size_t required_bits) {
     if (ba->capacity < required_bits) {
         size_t new_capacity = ba->capacity * 2;
         while (new_capacity < required_bits) {
             new_capacity *= 2;
         }
         
         size_t new_bytes = (new_capacity + 7) / 8;
         uint8_t *new_data = (uint8_t*)realloc(ba->data, new_bytes);
         if (new_data) {
             ba->data = new_data;
             ba->capacity = new_capacity;
             // 初始化新分配的内存
             memset(ba->data + (ba->bit_length + 7) / 8, 0, new_bytes - (ba->bit_length + 7) / 8);
             return true;
         } else {
             return false; // 内存分配失败
         }
     }
     return true;
 }
 
 // 添加一个位到位数组
 static inline bool hans_bitarray_append_bit(HANSBitArray *ba, int bit) {
     if (!hans_bitarray_ensure_capacity(ba, ba->bit_length + 1)) {
         return false;
     }
     
     if (bit) {
         ba->data[ba->bit_length / 8] |= (1 << (7 - (ba->bit_length % 8)));
     }
     
     ba->bit_length++;
     return true;
 }
 
 // 添加多个位到位数组
 static inline bool hans_bitarray_extend(HANSBitArray *ba, const char *bits) {
     size_t len = strlen(bits);
     if (!hans_bitarray_ensure_capacity(ba, ba->bit_length + len)) {
         return false;
     }
     
     for (size_t i = 0; i < len; i++) {
         if (bits[i] == '1') {
             ba->data[ba->bit_length / 8] |= (1 << (7 - (ba->bit_length % 8)));
         }
         ba->bit_length++;
     }
     return true;
 }
 
 // 释放位数组
 static inline void hans_bitarray_free(HANSBitArray *ba) {
     if (ba) {
         if (ba->data) {
             free(ba->data);
         }
         free(ba);
     }
 }
 
 // 比较函数用于排序
 static inline int hans_compare_freq(const void *a, const void *b) {
     uint32_t count_a = *((uint32_t*)a + 1);
     uint32_t count_b = *((uint32_t*)b + 1);
     
     // 降序排列
     if (count_a > count_b) return -1;
     if (count_a < count_b) return 1;
     return 0;
 }
 
 // 统计符号频率并排序
 static inline bool hans_get_symbol_freq(void *data, size_t size, int bit_width, uint32_t *freq_dict, uint32_t *freq_dict_reverse) {
     uint32_t *counts;
     int max_symbols = (bit_width == 8) ? HANS_MAX_SYMBOLS_8BIT : HANS_MAX_SYMBOLS_16BIT;
     
     // 分配计数数组
     counts = (uint32_t*)calloc(max_symbols, sizeof(uint32_t));
     if (!counts) {
         return false;
     }
     
     // 统计频率
     if (bit_width == 8) {
         uint8_t *bytes = (uint8_t*)data;
         for (size_t i = 0; i < size; i++) {
             counts[bytes[i]]++;
         }
     } else { // bit_width == 16
         uint16_t *words = (uint16_t*)data;
         for (size_t i = 0; i < size / 2; i++) {
             counts[words[i]]++;
         }
     }
     
     // 创建排序用的数组
     uint32_t (*freq_pairs)[2] = (uint32_t(*)[2])malloc(max_symbols * sizeof(*freq_pairs));
     if (!freq_pairs) {
         free(counts);
         return false;
     }
     
     // 填充频率对
     for (int i = 0; i < max_symbols; i++) {
         freq_pairs[i][0] = i;        // 符号
         freq_pairs[i][1] = counts[i]; // 频率
     }
     
     // 按频率排序
     qsort(freq_pairs, max_symbols, sizeof(*freq_pairs), hans_compare_freq);
     
     // 填充频率字典
     for (int i = 0; i < max_symbols; i++) {
         uint32_t symbol = freq_pairs[i][0];
         freq_dict[symbol] = i;
         freq_dict_reverse[i] = symbol;
     }
     
     free(freq_pairs);
     free(counts);
     return true;
 }
 
 // 压缩一个数据块
 static inline bool hans_compress_block(void *block_data, size_t block_size, uint32_t *freq_dict, 
                    int bit_width, HANSBitArray *compressed_data) {
     uint32_t *compressed_block = (uint32_t*)malloc(block_size * sizeof(uint32_t));
     if (!compressed_block) {
         return false;
     }
     
     // 将块中的每个值映射到其频率索引
     if (bit_width == 8) {
         uint8_t *bytes = (uint8_t*)block_data;
         for (size_t i = 0; i < block_size; i++) {
             compressed_block[i] = freq_dict[bytes[i]];
         }
     } else { // bit_width == 16
         uint16_t *words = (uint16_t*)block_data;
         for (size_t i = 0; i < block_size; i++) {
             compressed_block[i] = freq_dict[words[i]];
         }
     }
     
     // 找出最大值以确定位宽
     uint32_t max_val = 0;
     for (size_t i = 0; i < block_size; i++) {
         if (compressed_block[i] > max_val) {
             max_val = compressed_block[i];
         }
     }
     
     // 计算所需的位数
     int num_bits = 0;
     if (max_val > 0) {
         uint32_t temp = max_val;
         while (temp) {
             num_bits++;
             temp >>= 1;
         }
         num_bits = num_bits - 1; // 与原代码保持一致
     }
     
     // 添加元数据位
     int BITLEN_N = (bit_width == 8) ? HANS_BITLEN_N_8BIT : HANS_BITLEN_N_16BIT;
     char meta_bits[BITLEN_N + 1];
     for (int i = 0; i < BITLEN_N; i++) {
         meta_bits[i] = ((num_bits >> (BITLEN_N - 1 - i)) & 1) ? '1' : '0';
     }
     meta_bits[BITLEN_N] = '\0';
     if (!hans_bitarray_extend(compressed_data, meta_bits)) {
         free(compressed_block);
         return false;
     }
     
     // 添加压缩后的数据
     for (size_t i = 0; i < block_size; i++) {
         char value_bits[33]; // 最多32位 + 结束符
         for (int j = 0; j < num_bits + 1; j++) {
             value_bits[j] = ((compressed_block[i] >> (num_bits - j)) & 1) ? '1' : '0';
         }
         value_bits[num_bits + 1] = '\0';
         if (!hans_bitarray_extend(compressed_data, value_bits)) {
             free(compressed_block);
             return false;
         }
     }
     
     free(compressed_block);
     return true;
 }
 
 // 前向声明模板函数
 template <typename T>
 static inline bool h_HANS(long long& csize, uint8_t* in, uint8_t* out);
 
 /**
  * HANS压缩算法 - 8位版本
  * 
  * @param csize 输入/输出参数，输入时表示原数据大小，输出时表示压缩后大小
  * @param in 输入数据
  * @param out 输出缓冲区，用于存储压缩后的数据
  * @return 成功返回true，失败返回false
  */
 static inline bool h_HANS8(long long& csize, uint8_t* in, uint8_t* out) {
     return h_HANS<uint8_t>(csize, in, out);
 }
 
 /**
  * HANS压缩算法 - 16位版本
  * 
  * @param csize 输入/输出参数，输入时表示原数据大小，输出时表示压缩后大小
  * @param in 输入数据
  * @param out 输出缓冲区，用于存储压缩后的数据
  * @return 成功返回true，失败返回false
  */
 static inline bool h_HANS16(long long& csize, uint8_t* in, uint8_t* out) {
     return h_HANS<uint16_t>(csize, in, out);
 }
 
 /**
  * HANS压缩算法 - 通用模板
  * 
  * @param csize 输入/输出参数，输入时表示原数据大小，输出时表示压缩后大小
  * @param in 输入数据
  * @param out 输出缓冲区，用于存储压缩后的数据
  * @return 成功返回true，失败返回false
  */
 template <typename T>
 static inline bool h_HANS(long long& csize, uint8_t* in, uint8_t* out) {
     const int bit_width = sizeof(T) * 8;
     const int block_size = HANS_DEFAULT_BLOCK_SIZE;
     const int max_symbols = (bit_width == 8) ? HANS_MAX_SYMBOLS_8BIT : HANS_MAX_SYMBOLS_16BIT;
     
     if (bit_width != 8 && bit_width != 16) {
         return false;
     }
     
     // 保存原始大小
     long long original_size = csize;
     
     // 分配频率字典
     uint32_t *freq_dict = (uint32_t*)malloc(max_symbols * sizeof(uint32_t));
     uint32_t *freq_dict_reverse = (uint32_t*)malloc(max_symbols * sizeof(uint32_t));
     if (!freq_dict || !freq_dict_reverse) {
         if (freq_dict) free(freq_dict);
         if (freq_dict_reverse) free(freq_dict_reverse);
         return false;
     }
     
     // 构建频率表
     if (!hans_get_symbol_freq(in, original_size, bit_width, freq_dict, freq_dict_reverse)) {
         free(freq_dict);
         free(freq_dict_reverse);
         return false;
     }
     
     // 创建压缩数据结构
     HANSBitArray *compressed_data = hans_bitarray_create(original_size * bit_width); // 初始容量
     if (!compressed_data) {
         free(freq_dict);
         free(freq_dict_reverse);
         return false;
     }
     
     // 写入元素总数（64位）
     size_t total_elements = (bit_width == 8) ? original_size : original_size / 2;
     char total_bits[65];
     for (int i = 0; i < 64; i++) {
         total_bits[i] = ((total_elements >> (63 - i)) & 1) ? '1' : '0';
     }
     total_bits[64] = '\0';
     if (!hans_bitarray_extend(compressed_data, total_bits)) {
         hans_bitarray_free(compressed_data);
         free(freq_dict);
         free(freq_dict_reverse);
         return false;
     }
     
     // 计算块数
     size_t element_size = bit_width / 8;
     size_t num_elements = original_size / element_size;
     size_t num_blocks = num_elements / block_size;
     
     // 分块压缩
     for (size_t i = 0; i < num_blocks; i++) {
         void *block = (uint8_t*)in + i * block_size * element_size;
         if (!hans_compress_block(block, block_size, freq_dict, bit_width, compressed_data)) {
             hans_bitarray_free(compressed_data);
             free(freq_dict);
             free(freq_dict_reverse);
             return false;
         }
     }
     
     // 处理剩余数据
     size_t remaining = num_elements % block_size;
     if (remaining) {
         void *last_block = (uint8_t*)in + num_blocks * block_size * element_size;
         if (!hans_compress_block(last_block, remaining, freq_dict, bit_width, compressed_data)) {
             hans_bitarray_free(compressed_data);
             free(freq_dict);
             free(freq_dict_reverse);
             return false;
         }
     }
     
     // 复制压缩后的数据到输出缓冲区
     size_t compressed_bytes = (compressed_data->bit_length + 7) / 8;
     memcpy(out, compressed_data->data, compressed_bytes);
     
     // 更新压缩后的大小
     csize = (long long)compressed_bytes;
     
     // 释放资源
     free(freq_dict);
     free(freq_dict_reverse);
     hans_bitarray_free(compressed_data);
     
     return true;
 }
 
 #endif // H_HANS_H