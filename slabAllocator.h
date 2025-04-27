#ifndef SLAB_ALLOCATOR_H
#define SLAB_ALLOCATOR_H

#include <stdint.h>
#include <stdlib.h>

struct obj_header {
    struct obj_header *next;
};

struct slab {
    struct slab *next;
    void *memory;
    uint16_t free_objects;
    uint16_t total_objects;
    struct obj_header *free;
};

struct slab_cache {
    size_t obj_size;
    size_t slab_size;
    struct slab *slabs;
};

struct slab_cache* create_cache(size_t obj_size);
void* slab_alloc(struct slab_cache *cache);
void slab_free(struct slab_cache *cache, void *ptr);
struct slab* create_slab(struct slab_cache *cache);
void destroy_cache(struct slab_cache *cache);

#endif