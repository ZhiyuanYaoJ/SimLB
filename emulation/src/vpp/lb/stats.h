/*
 * Copyright (c) 2016 Cisco and/or its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Gathering stats for each AS and put them into shared memory
 */

#include <vnet/vnet.h>

#include <sys/mman.h>
#include <sys/stat.h>           /* For mode constants */
#include <fcntl.h>              /* For O_* constants */
#include "shm.h"

#define LB_DEFAULT_FLOW_TIMEOUT 40
#define U32_MAX 0xFFFFFFFF      /* Max value of 32-bit integer */
#define BIT_MAX 0x01 << (SHM_N_BIN - 1)

/**
 * @brief TCP flags
 */
#define TCP_FLAGS_NULL 0x00
#define TCP_FLAGS_SYN 0x02
#define TCP_FLAGS_ACK 0x10
// #define TCP_FLAGS_FINACK 0x11
#define TCP_FLAGS_RSTACK 0x14
#define TCP_FLAGS_PSHACK 0x18
#define TCP_FLAGS_PSH 0x08
#define TCP_FLAGS_RST 0x04
 
// #define LB_DEBUG

/**
 * @brief type of packet in the state machine
 */
typedef enum stat_packet_type
{
#define _(a,b) a,
	lb_foreach_stat_packet_type
#undef _
} stat_packet_type_t;

/**
 * Bitwise indexing
 */
#define GetBit(var, bit) ((var & ((u64)0x01 << (SHM_N_BIN - bit - 1))) != 0) // Returns true if bit is set
#define SetBit(var, bit) (var |= ((u64)0x01 << (SHM_N_BIN - bit - 1)))
#define MuteBit(var, bit) (var &= ~((u64)0x01 << (SHM_N_BIN - bit - 1)))
#define FlipBit(var, bit) (var ^= ((u64)0x01 << (SHM_N_BIN - bit - 1)))

/**
 * Check whether we should memcpy msg_out_cache to frames
 */
#define shm_vip_if_update_msg(t, t_last) \
        if (PREDICT_FALSE (t - t_last >= SHM_UPT_DT))

/*--- MACRO for lbhash ---*/
#define if_tsecr_valid(tsecr) if (PREDICT_TRUE(buffer->tsecr != U32_MAX && buffer->tsecr != 0))
#define if_flag_has_ack(flag) if (PREDICT_TRUE(flag & TCP_FLAGS_ACK))
#define if_flag_is_syn(flag) if (PREDICT_TRUE(flag == TCP_FLAGS_SYN))
#define if_flag_has_rst(flag) if (PREDICT_FALSE(flag & TCP_FLAGS_RST))
#define if_flag_has_psh(flag) if (PREDICT_FALSE(flag & TCP_FLAGS_PSH))
#define if_flag_not_only_ack(flag) if (PREDICT_FALSE(flag & ~TCP_FLAGS_ACK))
#define if_flag_not_null(flag) if (PREDICT_TRUE (flag != TCP_FLAGS_NULL))
#define if_flag_not_null_pf(flag) if (PREDICT_FALSE (flag != TCP_FLAGS_NULL))
#define if_flag_is_rst(flag) if (PREDICT_TRUE(flag == TCP_FLAGS_RST))
#define if_inconsistent(bucket, index, hash) \
        if (PREDICT_FALSE( \
        (bucket->tcp_flag[index] != TCP_FLAGS_NULL) \
        && (hash != bucket->hash[index]) \
        ))
#define register_new_flow(sym) \
        ({ \
        bucket->t_init[index] = buffer->time_now; \
        packet_type = SPT_##sym; \
        })
#define register_reservoir_as(ptr, sym, key, value) \
        ptr->sym[key] = value;
#define update_stat_cnt(cnt, sym) \
        ({ \
        packet_type = SPT_##cnt##_##sym; \
        })


#define _(a,b,c,d,e) a b; 
#define __(a,b,c,d,e) a b[c];
#define _construct(a) \
typedef struct { \
        lb_foreach_##a \
} __attribute__((packed)) a##_t;

// construct required structs
lb_foreach_typedef_struct;

#undef _
#undef __
#undef ___

#define _(a,b,c,d,e) a * b; 

typedef struct {
        
        /**
         * Ptr to the each field in the shm layout
         */
        lb_foreach_layout
        
        /**
         * msg_in sequence id
         */
        u32 *id_in;

        /**
         * msg_out sequence id
         */
        u32 id_out;

        /**
         * Starting point of the shared memory
         */
        char *mem;

        /**
         * fd
         */
        int fd;

        /**
         * Filename of the shared memory
         */
        char name[16];
} lb_vip_shm_t;

#undef _

extern stat_buffer_t stat_buffer;

clib_error_t * shm_vip_init_mem(lb_vip_shm_t *lbshm, const u32 id);
clib_error_t * shm_vip_del_mem(lb_vip_shm_t *lbshm);
void shm_as_clear_cache(lb_vip_shm_t *shm, const u32 id);
as_stat_t * shm_get_as_stat(lb_vip_shm_t *lbshm, const u32 id);
reservoir_as_t * shm_get_as_reservoir(lb_vip_shm_t *lbshm, const u32 id);
float * shm_get_as_score(lb_vip_shm_t *lbshm, const u32 id);
void shm_refresh_stat_buffer(stat_buffer_t *sb);
void shm_memcpy_frame_out(lb_vip_shm_t *lbshm, f32 time_now);
void shm_memcpy_frame_in(lb_vip_shm_t *lbshm);
