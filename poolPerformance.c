#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "poolAllocator.h"

#define BATCH_SIZE 32
#define INPUT_DIM 784
#define HIDDEN_DIM 128
#define OUTPUT_DIM 10

typedef struct {
    void* data;
    int rows;
    int cols;
} Tensor;

float W1_data[INPUT_DIM * HIDDEN_DIM];
float b1_data[HIDDEN_DIM];
float W2_data[HIDDEN_DIM * OUTPUT_DIM];
float b2_data[OUTPUT_DIM];
Tensor W1;
Tensor W2;

struct memory_pool* tensor_pool = NULL;

void init_pool_system() {
    size_t max_size = sizeof(float) * BATCH_SIZE * 
                     (INPUT_DIM > HIDDEN_DIM ? INPUT_DIM : HIDDEN_DIM);

    tensor_pool = pool_create(max_size, 10);
    if (!tensor_pool) {
        printf("Failed to create memory pool!\n");
        exit(1);
    }
}

void free_tensor(Tensor* t) {
    if (t && t->data) {
        free(t->data);
        t->data = NULL;
    }
}

void free_pool_tensor(Tensor* t) {
    if (t && t->data) {
        pool_free(tensor_pool, t->data);
        t->data = NULL;
    }
}

void matmul(const Tensor* A, const Tensor* B, Tensor* out) {
    float* a = (float*)A->data;
    float* b = (float*)B->data;
    float* c = (float*)out->data;
    
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            c[i * out->cols + j] = 0;
            for (int k = 0; k < A->cols; k++) {
                c[i * out->cols + j] += a[i * A->cols + k] * b[k * B->cols + j];
            }
        }
    }
}

void add_bias(Tensor* out, const float* bias) {
    float* data = (float*)out->data;
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            data[i * out->cols + j] += bias[j];
        }
    }
}

void relu(Tensor* t) {
    float* data = (float*)t->data;
    int size = t->rows * t->cols;
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}

void run_standard_allocator() {
    Tensor input = {malloc(sizeof(float)*BATCH_SIZE*INPUT_DIM), BATCH_SIZE, INPUT_DIM};
    Tensor h1 = {malloc(sizeof(float)*BATCH_SIZE*HIDDEN_DIM), BATCH_SIZE, HIDDEN_DIM};
    Tensor h2 = {malloc(sizeof(float)*BATCH_SIZE*HIDDEN_DIM), BATCH_SIZE, HIDDEN_DIM};
    Tensor h3 = {malloc(sizeof(float)*BATCH_SIZE*HIDDEN_DIM), BATCH_SIZE, HIDDEN_DIM};
    Tensor h4 = {malloc(sizeof(float)*BATCH_SIZE*HIDDEN_DIM), BATCH_SIZE, HIDDEN_DIM};
    Tensor output = {malloc(sizeof(float)*BATCH_SIZE*OUTPUT_DIM), BATCH_SIZE, OUTPUT_DIM};
    
    for (int i = 0; i < BATCH_SIZE*INPUT_DIM; i++) {
        ((float*)input.data)[i] = -5.0f + ((float)rand() / (float)RAND_MAX) * 10.0f;
    }
    
    matmul(&input, &W1, &h1);
    add_bias(&h1, b1_data);
    relu(&h1);
    
    matmul(&h1, &W1, &h2);
    add_bias(&h2, b1_data);
    relu(&h2);
    
    matmul(&h2, &W1, &h3);
    add_bias(&h3, b1_data);
    relu(&h3);
    
    matmul(&h3, &W1, &h4);
    add_bias(&h4, b1_data);
    relu(&h4);
    
    matmul(&h4, &W2, &output);
    add_bias(&output, b2_data);
    
    free_tensor(&input);
    free_tensor(&h1);
    free_tensor(&h2);
    free_tensor(&h3);
    free_tensor(&h4);
    free_tensor(&output);
}

void run_pool_allocator() {
    Tensor input = {pool_alloc(tensor_pool), BATCH_SIZE, INPUT_DIM};
    Tensor h1 = {pool_alloc(tensor_pool), BATCH_SIZE, HIDDEN_DIM};
    Tensor h2 = {pool_alloc(tensor_pool), BATCH_SIZE, HIDDEN_DIM};
    Tensor h3 = {pool_alloc(tensor_pool), BATCH_SIZE, HIDDEN_DIM};
    Tensor h4 = {pool_alloc(tensor_pool), BATCH_SIZE, HIDDEN_DIM};
    Tensor output = {pool_alloc(tensor_pool), BATCH_SIZE, OUTPUT_DIM};
    
    for (int i = 0; i < BATCH_SIZE*INPUT_DIM; i++) {
        ((float*)input.data)[i] = -5.0f + ((float)rand() / (float)RAND_MAX) * 10.0f;
    }
    
    matmul(&input, &W1, &h1);
    add_bias(&h1, b1_data);
    relu(&h1);
    
    matmul(&h1, &W1, &h2);
    add_bias(&h2, b1_data);
    relu(&h2);
    
    matmul(&h2, &W1, &h3);
    add_bias(&h3, b1_data);
    relu(&h3);
    
    matmul(&h3, &W1, &h4);
    add_bias(&h4, b1_data);
    relu(&h4);
    
    matmul(&h4, &W2, &output);
    add_bias(&output, b2_data);
    
    free_pool_tensor(&input);
    free_pool_tensor(&h1);
    free_pool_tensor(&h2);
    free_pool_tensor(&h3);
    free_pool_tensor(&h4);
    free_pool_tensor(&output);
}

void clear_cpu_cache() {
    int* cache_clear = (int*)malloc(32 * 1024 * 1024);
    if (cache_clear) {
        for (int j = 0; j < 8 * 1024 * 1024; j++) {
            cache_clear[j] = j;
        }
        free(cache_clear);
    }
}

int main() {
    srand(time(NULL));
    
    // Initialize weight matrices
    for(int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++) {
        W1_data[i] = -0.5f + ((float)rand() / (float)RAND_MAX);
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
        b1_data[i] = -0.5f + ((float)rand() / (float)RAND_MAX);
    }
    for(int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++) {
        W2_data[i] = -0.5f + ((float)rand() / (float)RAND_MAX);
    }
    for(int i = 0; i < OUTPUT_DIM; i++) {
        b2_data[i] = -0.5f + ((float)rand() / (float)RAND_MAX);
    }
    
    W1.data = W1_data;
    W1.rows = INPUT_DIM;
    W1.cols = HIDDEN_DIM;
    
    W2.data = W2_data;
    W2.rows = HIDDEN_DIM;
    W2.cols = OUTPUT_DIM;
    
    init_pool_system();
    
    printf("Running benchmarks...\n");
    
    const int NUM_ITERATIONS = 100;
    double std_total = 0.0, pool_total = 0.0;
    clock_t start, end;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        clear_cpu_cache();
        start = clock();
        run_standard_allocator();
        end = clock();
        double std_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        std_total += std_time;
        printf("Standard allocator took %f seconds\n", std_time);
        
        clear_cpu_cache();
        start = clock();
        run_pool_allocator();
        end = clock();
        double pool_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        pool_total += pool_time;
        printf("Pool allocator took %f seconds\n", pool_time);
        
        // Reset the pool to ensure fair comparison in each iteration
        pool_reset(tensor_pool);
        
        usleep(1000);
    }
    
    printf("\n--- BENCHMARK RESULTS (%d iterations) ---\n", NUM_ITERATIONS);
    printf("Standard allocator average: %f seconds\n", std_total / NUM_ITERATIONS);
    printf("Pool allocator average: %f seconds\n", pool_total / NUM_ITERATIONS);
    
    double improvement = 100.0 * (std_total - pool_total) / std_total;
    printf("Improvement: %.2f%%\n", improvement);
    
    pool_destroy(tensor_pool);
    
    return 0;
}