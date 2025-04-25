#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#define INPUT_DIM 4
#define HIDDEN_DIM 5
#define OUTPUT_DIM 3

#define BATCH_SIZE 2

#define ARENA_SIZE (1024*1024 * 10)//just a number lol 10MB 
typedef struct {
float* data;
int rows;
int cols;

} Tensor;

void matmul(const Tensor* A, const Tensor* B, Tensor* out);


void add_bias(Tensor* out, const float* bias);

void relu(Tensor *t);


typedef struct {
void* memory;

size_t total_size;
size_t used;

int initialized;


} Arena;


#endif 