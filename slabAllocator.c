// Slab Memory Layout:
// [Object 1] -> [Object 2] -> [Object 3] -> ... -> [Object N] -> NULL

// Where each object looks like:
// [obj_header (next pointer)] [actual object data]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* Slab states */
#define SLAB_FULL      0
#define SLAB_PARTIAL   1
#define SLAB_EMPTY     2

struct obj_header {
    struct obj_header *next;  
};

struct slab {
    struct slab *next;        
    void *memory;           
    uint16_t free_objects;   
    uint16_t total_objects; 
    struct obj_header *free; 
    uint8_t state;        
};

struct slab_cache {
    size_t obj_size;        
    size_t slab_size;       
    struct slab *full;     
    struct slab *partial;  
    struct slab *empty;     
};

struct slab_cache* create_cache(size_t obj_size);
void* slab_alloc(struct slab_cache *cache);
void slab_free(struct slab_cache *cache, void *ptr);
struct slab* create_slab(struct slab_cache *cache);
void destroy_cache(struct slab_cache *cache);