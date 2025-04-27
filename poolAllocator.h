#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>

 typedef struct  {
    struct block_header* next;
}block_header;

typedef struct {
    void* memory;
    struct block_header* free_list;
    size_t block_size;
    size_t total_blocks;
    size_t free_blocks;
} memory_pool;

struct memory_pool* pool_create(size_t block_size, size_t num_blocks);
void* pool_alloc(struct memory_pool* pool);
void pool_free(struct memory_pool* pool, void* ptr);
void pool_reset(struct memory_pool* pool);
void pool_destroy(struct memory_pool* pool);


#endif