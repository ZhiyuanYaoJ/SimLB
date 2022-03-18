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

#include <lb/stats.h>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define _(a,b,c,d,e) e,

const static as_stat_t as_stat_default = { lb_foreach_as_stat };
const static alias_t alias_default = { lb_foreach_alias };

#undef _

/* variables */
stat_buffer_t stat_buffer;

inline
as_stat_t * 
shm_get_as_stat(lb_vip_shm_t *lbshm, const u32 id) {
        return (as_stat_t *) (&lbshm->msg_out_cache->body[id]);
}

inline
reservoir_as_t * 
shm_get_as_reservoir(lb_vip_shm_t *lbshm, const u32 id) {
        return lbshm->res_as + id;
}

inline
alias_t * 
shm_get_as_weight(lb_vip_shm_t *lbshm, const u32 id) {
        return (alias_t *) (&lbshm->msg_in_cache->weights[id]);
}

/**
 * Clear AS's cache in shared memory
 */
void
shm_as_clear_cache(lb_vip_shm_t *shm, const u32 id)
{
        memcpy (shm_get_as_stat(shm, id), &as_stat_default, sizeof(as_stat_t));
        memcpy (shm_get_as_weight(shm, id), &alias_default, sizeof(alias_t));
        shm->msg_in_cache->score[id] = 0;
        MuteBit (shm->msg_out_cache->b_header, id);
}

clib_error_t *
shm_vip_init_mem(lb_vip_shm_t *lbshm, const u32 id)
{

        sprintf(lbshm->name, "shm_vip_%u", id);
	if ((lbshm->fd = shm_open(lbshm->name, O_RDWR | O_CREAT, 0777)) < 0) {
		perror("shm_open");
                clib_warning("@lb_vip_init: shm_open error");
		return NULL;
	}
        if (ftruncate(lbshm->fd, SHM_SIZE) < 0) {
		perror("ftruncate");
                clib_warning("@lb_vip_init: ftruncate error");
		return NULL;
	}
	if ((lbshm->mem = mmap(0, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, 
                        lbshm->fd, 0)) == MAP_FAILED) {
		perror("mmap");
                clib_warning("@lb_vip_init: mmap error");
		return NULL;
        }

        char * ptr = lbshm->mem + SHM_OFFSET;

#define _(a,b,c,d,e) \
lbshm->b = (a *) ptr; \
ptr = (char *) ((a *) ptr + c);

        lb_foreach_layout

#undef _

        // clib_warning("@lbshm: init lbshm->name %s, lbshm->n_as @%p, lbshm->msg_out_cache @%p lbshm->msg_out_cache->b_header @%p", 
        //         lbshm->name, 
        //         lbshm->n_as,
        //         lbshm->msg_out_cache,
        //         &(lbshm->msg_out_cache->b_header));
        *lbshm->n_as = SHM_N_BIN;
        lbshm->msg_in_cache->id = 0;

        /* initialize counters */
        lbshm->id_out = 0;
        lbshm->id_in = (u32 *) lbshm->msg_in_cache;
        *lbshm->id_in = 0;
        
        clib_warning("@lb:stats shm_vip_init_mem vip:%u filename:%s\n", id, lbshm->name);
        return NULL;
}

clib_error_t *
shm_vip_del_mem(lb_vip_shm_t *lbshm)
{
        if (munmap (lbshm->mem, SHM_SIZE) < 0)
        {
                perror ("munmap");
                clib_warning("@lb_vip_del: munmap error");
                return NULL;
        }

        if (close (lbshm->fd) < 0)
        {
                perror ("close");
                clib_warning("@lb_vip_del: close error");
                return NULL;
        }

        if (shm_unlink (lbshm->name))
        {
                perror ("shm_unlink");
                clib_warning("@lb_vip_del: shm_unlink error");
                return NULL;
        }

        clib_warning("@lb:stats shm_vip_del_mem\n");

        return NULL;
}

inline
void 
shm_memcpy_frame_out(lb_vip_shm_t *lbshm, f32 time_now) {
        u32 seq_id = ++lbshm->id_out;
        // clib_warning ("@lbshm: memcpy frame out sid = %u\n", seq_id);
        msg_out_t *frame_trg = lbshm->msg_out_frames + (seq_id & SHM_FRAME_MASK);
        /* since msg_out_cache id is always zero, automatically get lock */
        lbshm->msg_out_cache->ts = time_now;
        memcpy(frame_trg, lbshm->msg_out_cache, sizeof(msg_out_t));
        /* put lock by updating the frame_trg id in the end */
        frame_trg->id = seq_id;
        // clib_warning ("@lbshm: lbshm->n_as = %p, delta frame_trg-n_as = %p\n", lbshm->n_as, (void *) frame_trg - (void *) lbshm->n_as);
}

inline
void 
shm_memcpy_frame_in(lb_vip_shm_t *lbshm) {
        u32 seq_id_base = *lbshm->id_in;
        msg_in_t *frame_trg = lbshm->msg_in_frames + ((seq_id_base + 1) & SHM_FRAME_MASK);
        msg_in_t *frame_trg_base = frame_trg;
        u32 seq_id = frame_trg->id;
        // clib_warning("@lb:frame_in:first: old seq_id %u next seq_id %u", *lbshm->id_in, seq_id);
        while (seq_id > seq_id_base) {
                /* n-frame buffer ids are sorted */
                seq_id_base = seq_id;
                frame_trg_base = frame_trg;
                frame_trg = lbshm->msg_in_frames + ((seq_id+1) & SHM_FRAME_MASK);
                seq_id = frame_trg->id;
                // clib_warning("@lb:frame_in: old seq_id %u next seq_id %u", seq_id_base, seq_id);
        }
        // clib_warning("@lb:frame_in:final sizeof(msg_in_t)=%u, msg_in old seq_id %u, new seq_id %u", sizeof(msg_in_t), *lbshm->id_in, seq_id_base);
        if (PREDICT_TRUE(seq_id_base > *lbshm->id_in)) {
                memcpy(lbshm->msg_in_cache, frame_trg_base, sizeof(msg_in_t));
                // clib_warning("@lb:frame_in: memcpy!");
        }
}