/*
 * Copyright (c) 2012 Cisco and/or its affiliates.
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
 * vppinfra already includes tons of different hash tables.
 * MagLev flow table is a bit different. It has to be very efficient
 * for both writing and reading operations. But it does not need to
 * be 100% reliable (write can fail). It also needs to recycle
 * old entries in a lazy way.
 *
 * This hash table is the most dummy hash table you can do.
 * Fixed total size, fixed bucket size.
 * Advantage is that it could be very efficient (maybe).
 *
 */

#ifndef LB_PLUGIN_LB_LBHASH_H_
#define LB_PLUGIN_LB_LBHASH_H_

#include <vnet/vnet.h>
#include <vppinfra/lb_hash_hash.h>
#include <lb/stats.h>

#if defined (__SSE4_2__)
#include <immintrin.h>
#endif

/*
 * @brief Number of entries per bucket.
 */
#define LBHASH_ENTRY_PER_BUCKET 4

#define LB_HASH_DO_NOT_USE_SSE_BUCKETS 0

/*
 * @brief One bucket contains 4 entries.
 * Each bucket takes one 64B cache line in memory.
 */
typedef struct {
  CLIB_CACHE_LINE_ALIGN_MARK (cacheline0);
  u32 hash[LBHASH_ENTRY_PER_BUCKET];
  u32 timeout[LBHASH_ENTRY_PER_BUCKET];
  u32 vip[LBHASH_ENTRY_PER_BUCKET];
  u32 value[LBHASH_ENTRY_PER_BUCKET];
#ifdef LB_STATS
  f32 t_init[LBHASH_ENTRY_PER_BUCKET]; /* initial timestamp */
  u8 tcp_flag[LBHASH_ENTRY_PER_BUCKET]; /* tcp flags */
#endif // LB_STATS
} lb_hash_bucket_t;

typedef struct {
  u32 buckets_mask;
  u32 timeout;
  lb_hash_bucket_t buckets[];
} lb_hash_t;

#define lb_hash_nbuckets(h) (((h)->buckets_mask) + 1)
#define lb_hash_size(h) ((h)->buckets_mask + LBHASH_ENTRY_PER_BUCKET)

#define lb_hash_foreach_bucket(h, bucket) \
  for (bucket = (h)->buckets; \
	bucket < (h)->buckets + lb_hash_nbuckets(h); \
	bucket++)

#define lb_hash_foreach_entry(h, bucket, i) \
    lb_hash_foreach_bucket(h, bucket) \
      for (i = 0; i < LBHASH_ENTRY_PER_BUCKET; i++)

#define lb_hash_foreach_valid_entry(h, bucket, i, now) \
    lb_hash_foreach_entry(h, bucket, i) \
       if (!clib_u32_loop_gt((now), bucket->timeout[i]))

#ifdef LB_STATS
/* For established flows | state in {SYNed, ACKed, PSHACKed} */
static_always_inline void
bucket_stat_update_get(u32 hash, lb_hash_bucket_t *bucket, u32 index, u32 time_now_sec, stat_buffer_t *buffer, lb_vip_shm_t *vip_shm, const u32 found_value)
{
  as_stat_t *stat = shm_get_as_stat(vip_shm, found_value);
  reservoir_as_t *reservoir_as = shm_get_as_reservoir(vip_shm, found_value);
  if_inconsistent (bucket, index, hash)
  {
#ifdef LB_DEBUG
    clib_warning("@lbshm-get: bucket_stat_update_getcls bucket: %p, index: %u, value(asid): %u, prev_state: %02x, current_flag: %02x, ack_num: %u, time: %.4f, stat->n_flow: %u, stat->n_fct: %u, packet_type: %u, hash: %x, buffer->d_n_flow %d, stat->n_flow_on %d", bucket, index, stat->as_index, bucket->tcp_flag[index], buffer->tcp_flag, buffer->tcp_ack, buffer->time_now, stat->n_flow, stat->n_fct, SPT_WEIRD, bucket->hash[index], buffer->d_n_flow, stat->n_flow_on);
#endif // LB_DEBUG
    return;
  }

  /* Get tcp flags */
  u8 tcp_flag = buffer->tcp_flag;
  u8 tcp_flag_prev = bucket->tcp_flag[index];
  /* Initialize packet type */
  u8 packet_type = SPT_NORM;
  /* Initialize reservoir sampling id */
  u8 res_id = rand() % RESERVOIR_N_BIN;
  tv_pair_f_t tv_buffer_f = {buffer->time_now, 0.};


  if_flag_has_ack (tcp_flag)        // current_flag & ACK == TRUE
  {
    if_flag_not_only_ack (tcp_flag)
    {
      if_flag_has_rst (tcp_flag)           // a) current_flag == RSTACK
      {
        bucket->timeout[index] = time_now_sec - 1; // evict the bucket for this flow by updating its timeout
        packet_type = SPT_FIRST_FIN;
        buffer->d_n_flow = -1;
        // put flow duration into reservoir sampling memory
        tv_buffer_f.v = buffer->time_now - bucket->t_init[index];  // get flow complete time
        register_reservoir_as(reservoir_as, fct, res_id, tv_buffer_f);
        // DEV: debug GSQ2
        // clib_warning("@lbshm-get-current-flag-is-rstack: bucket_stat_update_getcls bucket: %p, index: %u, value(asid): %u, prev_state: %02x, current_flag: %02x, time: %.4f, hash: %x, buffer->d_n_flow %d, stat->n_flow_on %d", bucket, index, stat->as_index, bucket->tcp_flag[index], buffer->tcp_flag, buffer->time_now, bucket->hash[index], buffer->d_n_flow, stat->n_flow_on);
      }
    }
    else                                  // c) current_flag == ACK
    {
      if_flag_has_ack (tcp_flag_prev)     // 2) / 3) state == ACKed / PSHACKed
      {
        tv_buffer_f.v = buffer->time_now - bucket->t_init[index]; // flow duration
        register_reservoir_as(reservoir_as, flow_duration, res_id, tv_buffer_f);          
      }
      else  // 1) state == SYNed (first ack)
      {
        /* update packet type */
        packet_type = SPT_FIRST_ACK;
        buffer->d_n_flow = 1;
        // DEV: debug GSQ2
        // clib_warning("@lbshm-get-first-ack: bucket_stat_update_getcls bucket: %p, index: %u, value(asid): %u, prev_state: %02x, current_flag: %02x, time: %.4f, hash: %x, buffer->d_n_flow %d, stat->n_flow_on %d", bucket, index, stat->as_index, bucket->tcp_flag[index], buffer->tcp_flag, buffer->time_now, bucket->hash[index], buffer->d_n_flow, stat->n_flow_on);
      }
    }
  }
  else if_flag_is_syn (tcp_flag)          // d) current_flag == SYN
  {
    update_stat_cnt (rtr, SYN);
  }
  else if_flag_is_rst (tcp_flag)          // e) current_flag == RST
  {
    update_stat_cnt (rtr, RST);
  }
  else                                    // f) current_flag beyond scope
  {
    packet_type = SPT_BEYOND_SCOPE;
  }

  /* for every packet */
  if (PREDICT_TRUE(packet_type < SPT_rtr_SYN))
  {
    /* if not weird, update state */
    bucket->tcp_flag[index] = tcp_flag;
  }
  stat->n_flow_on += buffer->d_n_flow;
  
#ifdef LB_DEBUG
  clib_warning("@lbshm-get: bucket: %p, index: %u, value(asid): %u, prev_state: %02x, current_flag: %02x, ack_num: %u, time: %.4f, stat->n_flow: %u, stat->n_fct: %u, packet_type: %u, hash: %x, buffer->d_n_flow %d, stat->n_flow_on %d", bucket, index, stat->as_index, tcp_flag_prev, tcp_flag, buffer->tcp_ack, buffer->time_now, stat->n_flow, stat->n_fct, packet_type, bucket->hash[index], buffer->d_n_flow, stat->n_flow_on);
#endif // LB_DEBUG
}

/* For newly entered flows | state == IDLE */
static_always_inline void
bucket_stat_update_put(u32 hash, lb_hash_bucket_t *bucket, u32 index, u32 time_now_sec, stat_buffer_t *buffer, lb_vip_shm_t *vip_shm, const u32 found_value)
{
  as_stat_t *stat = shm_get_as_stat(vip_shm, found_value);
  u8 res_id = rand() % RESERVOIR_N_BIN;
  tv_pair_f_t tv_buffer_f = {buffer->time_now, 0.};

  if (bucket->tcp_flag[index] != TCP_FLAGS_NULL && bucket->tcp_flag[index] != TCP_FLAGS_RSTACK) // last flow timeout-ed without complete
  {
    if (bucket->hash[index] == hash) // same timeout-ed flow
    {
      if (bucket->value[index] != found_value) // same source but switched AS, wrap up last flow
      {
        u32 value_last = bucket->value[index];
        // as_stat_t *stat_last = shm_get_as_stat(vip_shm, value_last);
        reservoir_as_t *reservoir_as_last = shm_get_as_reservoir(vip_shm, value_last);
        tv_buffer_f.v = buffer->time_now - bucket->t_init[index] - LB_DEFAULT_FLOW_TIMEOUT;  /* guess flow complete time */
        register_reservoir_as(reservoir_as_last, fct, res_id, tv_buffer_f);
        // stat_last->n_flow_on--;
        // DEV: debug GSQ2
        // clib_warning("@lbshm-get: bucket_stat_update_put - same source but switched AS wraps up last flow, bucket: %p, index: %u, value(asid): %u, prev_state: %02x, current_flag: %02x, time: %.4f, hash: %x, previous hash: %x stat->n_flow_on %d", bucket, index, stat->as_index, bucket->tcp_flag[index], buffer->tcp_flag, buffer->time_now, hash, bucket->hash[index], stat->n_flow_on);
      }
      else // same source and same AS, no process is required
      {
        // DEV: debug GSQ2
        // clib_warning("@lbshm-get: bucket_stat_update_put - same source and same AS wraps up last flow, bucket: %p, index: %u, value(asid): %u, prev_state: %02x, current_flag: %02x, time: %.4f, hash: %x, previous hash: %x stat->n_flow_on %d", bucket, index, stat->as_index, bucket->tcp_flag[index], buffer->tcp_flag, buffer->time_now, hash, bucket->hash[index], stat->n_flow_on);
        return;
      }
    }
    else // a new flow coming, wrap up last flow
    {
#ifdef LB_DEBUG
      clib_warning("@lbshm-put: bucket->tcp_flag[index]=%02x, buffer->tcp_flag=%02x", bucket->tcp_flag[index], buffer->tcp_flag);
#endif // LB_DEBUG
      u32 value_last = bucket->value[index];
      // as_stat_t *stat_last = shm_get_as_stat(vip_shm, value_last);
      reservoir_as_t *reservoir_as_last = shm_get_as_reservoir(vip_shm, value_last);
      tv_buffer_f.v = buffer->time_now - bucket->t_init[index] - LB_DEFAULT_FLOW_TIMEOUT;  /* guess flow complete time */
      register_reservoir_as(reservoir_as_last, fct, res_id, tv_buffer_f);
      // stat_last->n_flow_on--;
      // DEV: debug GSQ2
      // clib_warning("@lbshm-get: bucket_stat_update_put - branch where a new flow comes and wraps up last flow, bucket: %p, index: %u, value(asid): %u, prev_state: %02x, current_flag: %02x, time: %.4f, hash: %x, previous hash: %x stat->n_flow_on %d", bucket, index, stat->as_index, bucket->tcp_flag[index], buffer->tcp_flag, buffer->time_now, bucket->hash[index], bucket->hash[index], stat->n_flow_on);
    }
  }
  
  /* Get tcp flags */
  u8 tcp_flag = buffer->tcp_flag;
  // u8 tcp_flag_prev = bucket->tcp_flag[index];
  /* Initialize packet type */
  u8 packet_type = SPT_NORM;

  if_flag_is_syn (tcp_flag)          // d) current_flag == SYN
  {
    register_new_flow (FIRST_SYN);
  }
  else if_flag_is_rst (tcp_flag)          // e) current_flag == RST
  {
    bucket->timeout[index] = time_now_sec - 1; // evict the bucket for this flow by updating its timeout
  }
  else                                    // f) current_flag beyond scope
  {
    packet_type = SPT_BEYOND_SCOPE;
  }

  /* for every packet */
  if (PREDICT_TRUE(packet_type < SPT_rtr_SYN))
  {
    /* if not weird, update state */
    bucket->tcp_flag[index] = tcp_flag;
  }
  stat->n_flow_on += buffer->d_n_flow;

#ifdef LB_DEBUG
  clib_warning("@lbshm-put: bucket_stat_update_put bucket: %p, index: %u, value(asid): %u, prev_state: %02x, current_flag: %02x, ack_num: %u, time: %.4f, stat->n_flow: %u, stat->n_fct: %u, packet_type: %u, hash: %x, buffer->d_n_flow %d, stat->n_flow_on %d", bucket, index, stat->as_index, bucket->tcp_flag[index], tcp_flag, buffer->tcp_ack, buffer->time_now, stat->n_flow, stat->n_fct, packet_type, bucket->hash[index], buffer->d_n_flow, stat->n_flow_on);
#endif // LB_DEBUG
}
#endif // LB_STATS

static_always_inline
lb_hash_t *lb_hash_alloc(u32 buckets, u32 timeout)
{
  if (!is_pow2(buckets))
    return NULL;

  // Allocate 1 more bucket for prefetch
  u32 size = ((uword)&((lb_hash_t *)(0))->buckets[0]) +
      sizeof(lb_hash_bucket_t) * (buckets + 1);
  u8 *mem = 0;
  lb_hash_t *h;
  vec_alloc_aligned(mem, size, CLIB_CACHE_LINE_BYTES);
  clib_memset(mem, 0, size);
  h = (lb_hash_t *)mem;
  h->buckets_mask = (buckets - 1);
  h->timeout = timeout;
  return h;
}

static_always_inline
void lb_hash_free(lb_hash_t *h)
{
  u8 *mem = (u8 *)h;
  vec_free(mem);
}

static_always_inline
void lb_hash_prefetch_bucket(lb_hash_t *ht, u32 hash)
{
  lb_hash_bucket_t *bucket = &ht->buckets[hash & ht->buckets_mask];
  CLIB_PREFETCH(bucket, sizeof(*bucket), READ);
}

static_always_inline
#ifdef LB_STATS
void lb_hash_get(lb_hash_t *ht, u32 hash, u32 vip, u32 time_now,
		 u32 *available_index, u32 *found_value, stat_buffer_t *buffer, lb_vip_shm_t *vip_shm)
#else
void lb_hash_get(lb_hash_t *ht, u32 hash, u32 vip, u32 time_now,
		 u32 *available_index, u32 *found_value)
#endif // LB_STATS
{
  lb_hash_bucket_t *bucket = &ht->buckets[hash & ht->buckets_mask];
  *found_value = 0;
  *available_index = ~0;
#if __SSE4_2__ && LB_HASH_DO_NOT_USE_SSE_BUCKETS == 0
  u32 bitmask, found_index;
  __m128i mask;

  // mask[*] = timeout[*] > now
  mask = _mm_cmpgt_epi32(_mm_loadu_si128 ((__m128i *) bucket->timeout),
			 _mm_set1_epi32 (time_now));
  // bitmask[*] = now <= timeout[*/4]
  bitmask = (~_mm_movemask_epi8(mask)) & 0xffff;
  // Get first index with now <= timeout[*], if any.
  *available_index = (bitmask)?__builtin_ctz(bitmask)>>2:*available_index;

#ifdef LB_STATS
  /* if received is not a SYN packet, either go to the same bucket if a state is found, or "pretend" the bucket is full and assign according to hashing table */
  *available_index = (buffer->tcp_flag & TCP_FLAGS_SYN)?*available_index:~0;
#endif /* LB_STATS */

  // mask[*] = (timeout[*] > now) && (hash[*] == hash)
  mask = _mm_and_si128(mask,
		       _mm_cmpeq_epi32(
			   _mm_loadu_si128 ((__m128i *) bucket->hash),
			   _mm_set1_epi32 (hash)));

  // Load the array of vip values
  // mask[*] = (timeout[*] > now) && (hash[*] == hash) && (vip[*] == vip)
  mask = _mm_and_si128(mask,
		       _mm_cmpeq_epi32(
			   _mm_loadu_si128 ((__m128i *) bucket->vip),
			   _mm_set1_epi32 (vip)));

  // mask[*] = (timeout[*x4] > now) && (hash[*x4] == hash) && (vip[*x4] == vip)
  bitmask = _mm_movemask_epi8(mask);
  // Get first index, if any
  found_index = (bitmask)?__builtin_ctzll(bitmask)>>2:0;
  ASSERT(found_index < 4);
  *found_value = (bitmask)?bucket->value[found_index]:*found_value;
  bucket->timeout[found_index] =
      (bitmask)?time_now + ht->timeout:bucket->timeout[found_index];
#ifdef LB_STATS
  /* If found index of bucket */
  if (PREDICT_TRUE (bitmask))
  {
    bucket_stat_update_get(hash, bucket, found_index, time_now, buffer, vip_shm, *found_value);
  }
  // DEV: debug GSQ2
  // else
  // {
  //   clib_warning("@lbshm-get: bucket not found bucket: %p, index: %u, value(asid): %u, prev_state: %02x, current_flag: %02x, time: %.4f, hash: %x, previous hash: %x", bucket, found_index, *found_value, bucket->tcp_flag[found_index], buffer->tcp_flag, buffer->time_now, hash, bucket->hash[found_index]);
  // }
#endif // LB_STATS
#else
  u32 i;
  for (i = 0; i < LBHASH_ENTRY_PER_BUCKET; i++) {
      u8 cmp = (bucket->hash[i] == hash && bucket->vip[i] == vip);
      u8 timeouted = clib_u32_loop_gt(time_now, bucket->timeout[i]);
      *found_value = (cmp || timeouted)?*found_value:bucket->value[i];
      bucket->timeout[i] = (cmp || timeouted)?time_now + ht->timeout:bucket->timeout[i];
      *available_index = (timeouted && (*available_index == ~0))?i:*available_index;

      if (!cmp)
	return;
  }
#endif
}

static_always_inline
u32 lb_hash_available_value(lb_hash_t *h, u32 hash, u32 available_index)
{
  return h->buckets[hash & h->buckets_mask].value[available_index];
}

static_always_inline
#ifdef LB_STATS
void lb_hash_put(lb_hash_t *h, u32 hash, u32 value, u32 vip,
		 u32 available_index, u32 time_now, stat_buffer_t *buffer, lb_vip_shm_t *vip_shm)
#else
void lb_hash_put(lb_hash_t *h, u32 hash, u32 value, u32 vip,
		 u32 available_index, u32 time_now)
#endif // LB_STATS
{
  lb_hash_bucket_t *bucket = &h->buckets[hash & h->buckets_mask];
#ifdef LB_STATS
  bucket_stat_update_put (hash, bucket, available_index, time_now, buffer, vip_shm, value);
#endif // LB_STATS
  bucket->hash[available_index] = hash;
  bucket->value[available_index] = value;
  bucket->timeout[available_index] = time_now + h->timeout;
  bucket->vip[available_index] = vip;
}

static_always_inline
u32 lb_hash_elts(lb_hash_t *h, u32 time_now)
{
  u32 tot = 0;
  lb_hash_bucket_t *bucket;
  u32 i;
  lb_hash_foreach_valid_entry(h, bucket, i, time_now) {
    tot++;
  }
  return tot;
}

#endif /* LB_PLUGIN_LB_LBHASH_H_ */
