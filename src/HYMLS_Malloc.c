#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

static char tmpbuf[32768];
static size_t tmppos = 0;

typedef struct ptr_size_
{
    void *ptr;
    size_t size;
} ptr_size;

static ptr_size *ptrbuf = NULL;
static ptr_size *ptrbuf_end = NULL;
static size_t ptrbuf_size = 1;
static size_t total_size = 0;
static size_t max_total_size = 0;

static void dummy_add_ptr(void *ptr, size_t size)
{
}

static int dummy_del_ptr(void *ptr)
{
    return -1;
}

static void (*add_ptr)(void *ptr, size_t size) = dummy_add_ptr;
static int (*del_ptr)(void *ptr) = dummy_del_ptr;

#define _printf(fmt, ...) {                                \
        void (*old_add_ptr)(void *ptr, size_t size) = add_ptr;  \
        int (*old_del_ptr)(void *ptr) = del_ptr;                \
        add_ptr = dummy_add_ptr;                                \
        del_ptr = dummy_del_ptr;                                \
                                                                \
        fprintf(stderr, fmt, ##__VA_ARGS__);                    \
        fflush(stderr);                                         \
                                                                \
        add_ptr = old_add_ptr;                                  \
        del_ptr = old_del_ptr;                                  \
    }

static void* dummy_malloc(size_t size)
{
    if (tmppos + size >= sizeof(tmpbuf))
    {
        _printf("Not enough space. Requested %zu, %li available\n", size, sizeof(tmpbuf) - tmppos);
        exit(1);
    }
    void *retptr = tmpbuf + tmppos;
    tmppos += size;
    return retptr;
}

static void* dummy_calloc(size_t nmemb, size_t size)
{
    void *ptr = dummy_malloc(nmemb * size);
    char *ptr_it = (char *)ptr;
    size_t i = 0;
    for (; i < nmemb * size; ++i)
        *(ptr_it++) = '\0';
    return ptr;
}

static void *dummy_memalign(size_t alignment, size_t size)
{
    size_t pos = (size_t)tmpbuf + tmppos;
    while (pos % alignment)
    {
        pos++;
        tmppos++;
    }
    return dummy_malloc(size);
}

static void *dummy_aligned_alloc(size_t alignment, size_t size)
{
    return dummy_memalign(alignment, size);
}

static int dummy_posix_memalign(void** memptr, size_t alignment, size_t size)
{
    *memptr = dummy_memalign(alignment, size);
    return 0;
}

static void dummy_free(void *ptr)
{
}

static int malloc_initialized = 0;

static void* (*real_malloc)(size_t size) = dummy_malloc;
static void* (*real_calloc)(size_t nmemb, size_t size) = dummy_calloc;
static void* (*real_realloc)(void *ptr, size_t size) = NULL;
static void* (*real_memalign)(size_t alignment, size_t size) = dummy_memalign;
static void* (*real_valloc)(size_t size) = NULL;
static int   (*real_posix_memalign)(void** memptr, size_t alignment,
                                    size_t size) = dummy_posix_memalign;
static void  (*real_free)(void *ptr) = dummy_free;
static void* (*real_aligned_alloc)(size_t alignment, size_t size) = dummy_aligned_alloc;
static size_t(*real_malloc_usable_size)(void *ptr) = NULL;
static void* (*real_pvalloc)(size_t size) = NULL;

static void* (*temp_malloc)(size_t size) = NULL;
static void* (*temp_calloc)(size_t nmemb, size_t size) = NULL;
static void* (*temp_realloc)(void *ptr, size_t size) = NULL;
static void* (*temp_memalign)(size_t alignment, size_t size) = NULL;
static void* (*temp_valloc)(size_t size) = NULL;
static int   (*temp_posix_memalign)(void** memptr, size_t alignment,
                                    size_t size) = NULL;
static void  (*temp_free)(void *ptr) = NULL;
static void* (*temp_aligned_alloc)(size_t alignment, size_t size) = NULL;
static size_t(*temp_malloc_usable_size)(void *ptr) = NULL;
static void* (*temp_pvalloc)(size_t size) = NULL;

static int real_add_ptr_hash(void *ptr, size_t size)
{
    ptr_size *ptrbuf_iter = ptrbuf + ((size_t)ptr % ptrbuf_size);
    for (; ptrbuf_iter != ptrbuf_end; ++ptrbuf_iter)
    {
        if (ptrbuf_iter->ptr == ptr)
        {
            if (ptrbuf_iter->size == size)
                return 1;
            else
                _printf("Pointer of size %zu reallocated but not freed %p\n", size, ptr);
        }

        if (ptrbuf_iter->ptr == NULL)
        {
            ptrbuf_iter->ptr = ptr;
            ptrbuf_iter->size = size;
            return 0;
        }
    }
    return -1;
}

static void ptrbuf_realloc()
{
    ptrbuf_size *= 2;
    if (ptrbuf == NULL)
        ptrbuf_size = 20;

    ptr_size *tmp_ptrbuf = ptrbuf;

    // Protect calloc call since apparently this can in turn call memalign
    void (*old_add_ptr)(void *ptr, size_t size) = add_ptr;
    int (*old_del_ptr)(void *ptr) = del_ptr;
    add_ptr = dummy_add_ptr;
    del_ptr = dummy_del_ptr;

    ptrbuf = (ptr_size *)real_calloc(ptrbuf_size, sizeof(ptr_size));

    add_ptr = old_add_ptr;
    del_ptr = old_del_ptr;

    if (tmp_ptrbuf)
    {
        ptr_size *ptrbuf_iter = tmp_ptrbuf;
        for (; ptrbuf_iter != ptrbuf_end; ++ptrbuf_iter)
        {
            if (ptrbuf_iter->ptr)
            {
                if (real_add_ptr_hash(ptrbuf_iter->ptr, ptrbuf_iter->size) == -1)
                    ptrbuf_realloc();
            }
        }
        real_free(tmp_ptrbuf);
    }
    ptrbuf_end = ptrbuf + ptrbuf_size;
}

static void real_add_ptr(void *ptr, size_t size)
{
    int ret = real_add_ptr_hash(ptr, size);

    if (ret == 0)
    {
        total_size += size;
        max_total_size = total_size > max_total_size ? total_size : max_total_size;
    }

    if (ret >= 0)
        return;

    ptrbuf_realloc();

    real_add_ptr(ptr, size);
}

static int real_del_ptr(void *ptr)
{
    if (!ptr) return -1;

    ptr_size *ptrbuf_iter = ptrbuf + ((size_t)ptr % ptrbuf_size);
    for (; ptrbuf_iter != ptrbuf_end; ++ptrbuf_iter)
        if (ptrbuf_iter->ptr == ptr)
        {
            total_size -= ptrbuf_iter->size;
            ptrbuf_iter->ptr = NULL;
            ptrbuf_iter->size = 0;
            return 0;
        }
    return -1;
}

size_t get_memory_usage()
{
    return total_size;
}

size_t get_max_memory_usage()
{
    return max_total_size;
}

static void hookfns()
{
    malloc_initialized = 1;

    add_ptr = dummy_add_ptr;
    del_ptr = dummy_del_ptr;

    temp_malloc             = (void* (*)(size_t)) dlsym(RTLD_NEXT, "malloc");
    temp_calloc             = (void* (*)(size_t, size_t)) dlsym(RTLD_NEXT, "calloc");
    temp_realloc            = (void* (*)(void *, size_t)) dlsym(RTLD_NEXT, "realloc");
    temp_free               = (void (*)(void *)) dlsym(RTLD_NEXT, "free");
    temp_memalign           = (void* (*)(size_t, size_t)) dlsym(RTLD_NEXT, "memalign");
    temp_valloc             = (void* (*)(size_t)) dlsym(RTLD_NEXT, "valloc");
    temp_posix_memalign     = (int (*)(void **, size_t, size_t)) dlsym(RTLD_NEXT, "posix_memalign");
    temp_aligned_alloc      = (void* (*)(size_t , size_t)) dlsym(RTLD_NEXT, "aligned_alloc");
    temp_malloc_usable_size = (size_t (*)(void *)) dlsym(RTLD_NEXT, "malloc_usable_size");
    temp_pvalloc            = (void *(*)(size_t)) dlsym(RTLD_NEXT, "pvalloc");

    if (!temp_malloc || !temp_calloc || !temp_realloc || !temp_memalign ||
        !temp_valloc || !temp_posix_memalign || !temp_free || !temp_aligned_alloc ||
        !temp_malloc_usable_size || !temp_pvalloc)
    {
        _printf("Error in `dlsym`: %s\n", dlerror());
        exit(1);
    }

    real_malloc             = temp_malloc;
    real_calloc             = temp_calloc;
    real_realloc            = temp_realloc;
    real_free               = temp_free;
    real_memalign           = temp_memalign;
    real_valloc             = temp_valloc;
    real_posix_memalign     = temp_posix_memalign;
    real_aligned_alloc      = temp_aligned_alloc;
    real_malloc_usable_size = temp_malloc_usable_size;
    real_pvalloc            = temp_pvalloc;

    add_ptr = (void (*)(void *, size_t))real_add_ptr;
    del_ptr = (int (*)(void *))real_del_ptr;
}

void* malloc(size_t size)
{
    if (!malloc_initialized)
        hookfns();

    void *new_ptr = real_malloc(size);
    add_ptr(new_ptr, size);
    return new_ptr;
}

void* calloc(size_t nmemb, size_t size)
{
    if (!malloc_initialized)
        hookfns();

    void *new_ptr = real_calloc(nmemb, size);
    add_ptr(new_ptr, nmemb * size);
    return new_ptr;
}

void* realloc(void *ptr, size_t size)
{
    if (!malloc_initialized)
        hookfns();

    del_ptr(ptr);
    void *new_ptr = real_realloc(ptr, size);
    add_ptr(new_ptr, size);
    return new_ptr;
}

void free(void *ptr)
{
    if (!malloc_initialized)
        hookfns();

    if (del_ptr(ptr) == 0)
        real_free(ptr);
}

void* memalign(size_t alignment, size_t size)
{
    if (!malloc_initialized)
        hookfns();

    void *new_ptr = real_memalign(alignment, size);
    add_ptr(new_ptr, size);
    return new_ptr;
}

int posix_memalign(void** memptr, size_t alignment, size_t size)
{
    if (!malloc_initialized)
        hookfns();

    int ret = real_posix_memalign(memptr, alignment, size);
    add_ptr(*memptr, size);
    return ret;
}

void* valloc(size_t size)
{
    if (!malloc_initialized)
        hookfns();

    _printf("valloc not implemented: %zu\n", size);
    return real_valloc(size);
}

void* aligned_alloc(size_t alignment, size_t size)
{
    if (!malloc_initialized)
        hookfns();

    void *new_ptr = real_aligned_alloc(alignment, size);
    add_ptr(new_ptr, size);
    return new_ptr;
}

size_t malloc_usable_size(void *ptr)
{
    if (!malloc_initialized)
        hookfns();

    _printf("malloc_usable_size not implemented\n");
    return real_malloc_usable_size(ptr);
}

void* pvalloc(size_t size)
{
    if (!malloc_initialized)
        hookfns();

    _printf("pvalloc not implemented: %zu\n", size);
    return real_pvalloc(size);
}
