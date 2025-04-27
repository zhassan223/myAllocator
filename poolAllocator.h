#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H

#include <stddef.h>

struct block_header {
    struct block_header* next;
};

struct memory_pool {
    void* memory;
    size_t block_size;
    size_t total_blocks;
    size_t free_blocks;
    struct block_header* free_list;
};

struct memory_pool* pool_create(size_t block_size, size_t num_blocks);
void* pool_alloc(struct memory_pool* pool);
void pool_free(struct memory_pool* pool, void* ptr);
void pool_reset(struct memory_pool* pool);
void pool_destroy(struct memory_pool* pool);

#endif /* POOL_ALLOCATOR_H */