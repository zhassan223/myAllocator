#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#define INPUT_DIM 4
#define HIDDEN_DIM 5
#define OUTPUT_DIM 3

#define BATCH_SIZE 2

typedef struct {
float* data;
int rows;
int cols;

} Tensor;

void matmul(const Tensor* A, const Tensor* B, Tensor* out);


void add_bias(Tensor* out, const float* bias);

void relu(Tensor *t);
#endif 