#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "slabAllocator.h"

struct slab* create_slab(struct slab_cache *cache) {
    struct slab *slab = malloc(sizeof(struct slab));
    if (!slab) return NULL;

    slab->memory = malloc(cache->slab_size);
    if (!slab->memory) {
        free(slab);
        return NULL;
    }
    
    slab->total_objects = cache->slab_size / cache->obj_size;
    slab->free_objects = slab->total_objects;
    
    //free list set up 
    char *current = (char *)slab->memory;
    slab->free = (struct obj_header *)current;
    
    // link objs in mem together
    for (size_t i = 0; i < slab->total_objects - 1; i++) {
        struct obj_header *obj = (struct obj_header *)current;
        current += cache->obj_size;
        obj->next = (struct obj_header *)current;
    }
    ((struct obj_header *)current)->next = NULL;
    
    return slab;
}

struct slab_cache* create_cache(size_t obj_size) {
    struct slab_cache *cache = malloc(sizeof(struct slab_cache));
    if (!cache) return NULL;

    cache->obj_size = obj_size < sizeof(struct obj_header) ? 
                     sizeof(struct obj_header) : obj_size;
    
    if (obj_size > 1024) {
        cache->slab_size = 4096 * ((obj_size / 4096) + 1);
        printf("Using larger slab size: %zu bytes\n", cache->slab_size);
    } else {
        cache->slab_size = 4096;  
    }
    
    cache->slabs = NULL;
    return cache;
}

void* slab_alloc(struct slab_cache *cache) {
    struct slab *slab = cache->slabs;
    struct slab *prev = NULL;
    
    while (slab && slab->free_objects == 0) {
        prev = slab;
        slab = slab->next;
    }
    
    if (!slab) {
        slab = create_slab(cache);
        if (!slab) return NULL;
        
        slab->next = cache->slabs;
        cache->slabs = slab;
    }
    
    void *obj = slab->free;
    slab->free = slab->free->next;
    slab->free_objects--;
    
    return obj;
}

void slab_free(struct slab_cache *cache, void *ptr) {
    struct slab *slab = cache->slabs;
    
    while (slab) {
        if (ptr >= slab->memory && 
            ptr < (void*)((char*)slab->memory + cache->slab_size)) {
            break;
        }
        slab = slab->next;
    }
    
    if (!slab) return;  // Bad pointer
    
    // add to free list
    struct obj_header *obj = ptr;
    obj->next = slab->free;
    slab->free = obj;
    slab->free_objects++;
}

void destroy_cache(struct slab_cache *cache) {
    if (!cache) return;
    
    struct slab *slab = cache->slabs;
    while (slab) {
        struct slab *next = slab->next;
        free(slab->memory);
        free(slab);
        slab = next;
    }
    
    free(cache);
}