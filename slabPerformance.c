#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "slabAllocator.h"

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

struct slab_cache* tensor_cache = NULL;

void init_slab_system() {
    size_t max_size = sizeof(float) * BATCH_SIZE * 
                     (INPUT_DIM > HIDDEN_DIM ? INPUT_DIM : HIDDEN_DIM);
    
    tensor_cache = create_cache(max_size);
    if (!tensor_cache) {
        printf("Failed to create slab cache!\n");
        exit(1);
    }
}

void free_tensor(Tensor* t) {
    if (t && t->data) {
        free(t->data);
        t->data = NULL;
    }
}

void free_slab_tensor(Tensor* t) {
    if (t && t->data) {
        slab_free(tensor_cache, t->data);
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

void run_slab_allocator() {
    Tensor input = {slab_alloc(tensor_cache), BATCH_SIZE, INPUT_DIM};
    Tensor h1 = {slab_alloc(tensor_cache), BATCH_SIZE, HIDDEN_DIM};
    Tensor h2 = {slab_alloc(tensor_cache), BATCH_SIZE, HIDDEN_DIM};
    Tensor h3 = {slab_alloc(tensor_cache), BATCH_SIZE, HIDDEN_DIM};
    Tensor h4 = {slab_alloc(tensor_cache), BATCH_SIZE, HIDDEN_DIM};
    Tensor output = {slab_alloc(tensor_cache), BATCH_SIZE, OUTPUT_DIM};
    
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
    
    free_slab_tensor(&input);
    free_slab_tensor(&h1);
    free_slab_tensor(&h2);
    free_slab_tensor(&h3);
    free_slab_tensor(&h4);
    free_slab_tensor(&output);
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
    
    init_slab_system();
    
    printf("Running benchmarks...\n");
    
    const int NUM_ITERATIONS = 100;
    double std_total = 0.0, slab_total = 0.0;
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
        run_slab_allocator();
        end = clock();
        double slab_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        slab_total += slab_time;
        printf("Slab allocator took %f seconds\n", slab_time);
        
        usleep(1000);
    }
    
    printf("\n--- BENCHMARK RESULTS (%d iterations) ---\n", NUM_ITERATIONS);
    printf("Standard allocator average: %f seconds\n", std_total / NUM_ITERATIONS);
    printf("Slab allocator average: %f seconds\n", slab_total / NUM_ITERATIONS);
    
    double improvement = 100.0 * (std_total - slab_total) / std_total;
    printf("Improvement: %.2f%%\n", improvement);
    
    destroy_cache(tensor_cache);
    
    return 0;
}