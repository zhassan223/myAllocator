#include  <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "poolAllocator.h"

typedef struct memory_pool {
    void* memory;
    size_t block_size;
    size_t total_blocks;
    size_t free_blocks;
    struct block_header* free_list;
} memory_pool;

typedef struct block_header {
	struct block_header* next;
} block_header;


struct memory_pool* pool_create(size_t block_size, size_t num_blocks){
    if(block_size<sizeof(block_header)) block_size = sizeof(block_header);


    memory_pool* mem_pool = malloc(sizeof(memory_pool));
    if(!mem_pool) return NULL;



    size_t total_size = block_size * num_blocks;
    mem_pool->memory = malloc(total_size);
    if(!mem_pool->memory){
        free(mem_pool);
        return NULL;

    }

    mem_pool->block_size = block_size;
    mem_pool->total_blocks = num_blocks;
    mem_pool->free_blocks = num_blocks;


    mem_pool->free_list = mem_pool->memory;
    block_header* curr = mem_pool->free_list;


    for(size_t i=0;i<num_blocks-1;i++){

        block_header* next = (block_header*)((char*)curr + block_size);
        curr->next = next;
        curr = next;


    }
    curr->next = NULL;
    return mem_pool;
}

void* pool_alloc(struct memory_pool* pool){
    if(!pool||!(pool->free_blocks)) return NULL;

    block_header* block = pool->free_list;

    pool->free_list = block->next;

    pool->free_blocks--;

    return block;
}

void pool_free(memory_pool* pool, void* ptr){
    if(!pool||!ptr) return;



    if(ptr<pool->memory || ptr>= (void*)(char*)pool->memory + (pool->total_blocks * pool->block_size)) return;



    block_header* block = (block_header*) ptr;


    block->next = pool->free_list;

    pool->free_list = block; 


    pool->free_blocks++;
}

void pool_reset(memory_pool* pool) {
    if (!pool) return;
    
    pool->free_list = pool->memory;
    block_header* curr = pool->free_list;

    size_t block_count = pool->total_blocks * pool->block_size;
    
    for (size_t i = 0; i < block_count - 1; i++) {
        block_header* next = (block_header*)((char*)curr + pool->block_size);
        curr->next = next;
        curr = next;
    }
    curr->next = NULL;
    
    pool->free_blocks = block_count;
}

void pool_destory(memory_pool* pool){
    if(!pool) return; 

    if(pool->memory){
        free(pool->memory);
        pool->memory = NULL;


    }

    free(pool);
}