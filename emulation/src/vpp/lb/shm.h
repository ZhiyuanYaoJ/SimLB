#define LB_STATS
#define LB_LSQ

#define SHM_SIZE 1048576
#define SHM_OFFSET 42
#define SHM_N_BIN 64
#define SHM_N_FRAME 4
#define SHM_FRAME_MASK 3
#define VIP_ID 1
#define SHM_UPT_DT 0.2
#define RESERVOIR_N_BIN 128

#define lb_foreach_stat_buffer \
_(f32, time_now, 1, "Current timestamp", 0.0) \
_(u8, tcp_flag, 1, "SYN/ACK/RSTACK/RST... flags in tcp header", 0) \
_(int8_t, d_n_flow, 1, "Counter buffer used for counting number of flow", 0) 

#define lb_foreach_tv_pair_u \
_(f32, t, 1, "time", 0.0) \
_(u32, v, 1, "value", 0) 

#define lb_foreach_tv_pair_f \
_(f32, t, 1, "time", 0.0) \
_(f32, v, 1, "value", 0.0) 

#define lb_foreach_tv_pair \
_(f32, t, 1, "time", 0.0) \
_(int32_t, v, 1, "value", 0) 

#define lb_foreach_as_stat \
_(u32, as_index, 1, "AS index", 0) \
_(int32_t, n_flow_on, 1, "Instantaneous number of established flows", 0) 

#define lb_foreach_reservoir_as \
__(tv_pair_f_t, fct, 128, "A list of time-value pair storing flow complete time", 0) \
__(tv_pair_f_t, flow_duration, 128, "A list of time-value pair storing flow duration", 0) 

#define lb_foreach_alias \
_(f32, odd, 1, "Probability of choosing local bucket", 1.0) \
_(u32, alias, 1, "Alias of local bucket", 0) 

#define lb_foreach_msg_out \
_(u32, id, 1, "Sequence ID of output message", 0) \
_(f32, ts, 1, "Corresponding timestamp", 0.0) \
_(u64, b_header, 1, "Binary header indicating which ASs are active", 0) \
__(as_stat_t, body, 64, "An array of stats info for each AS", 0) 

#define lb_foreach_msg_in \
_(u32, id, 1, "Sequence ID of output message", 0) \
_(f32, ts, 1, "Corresponding timestamp", 0.0) \
__(f32, score, 64, "Allocated weights for each AS", 1.0) \
__(alias_t, weights, 64, "An array of alias info for each AS", 0) 

#define lb_foreach_typedef_struct \
_construct(stat_buffer) \
_construct(tv_pair_u) \
_construct(tv_pair_f) \
_construct(tv_pair) \
_construct(as_stat) \
_construct(reservoir_as) \
_construct(alias) \
_construct(msg_out) \
_construct(msg_in) \

#define lb_foreach_stat_packet_type \
_(SPT_NORM, "/** Normal packet (do not increment counter) */") \
_(SPT_FIRST_ACK, "/** Received first ACK */") \
_(SPT_FIRST_DATA, "/** Received ACK to first data packet **/") \
_(SPT_FIRST_SYN, "/** Received first SYN */") \
_(SPT_FIRST_FIN, "/** Received first FIN */") \
_(SPT_PSHACK, "/** Received PSHACK (http query) **/") \
_(SPT_rtr_SYN, "/** Packet retransmission (PSHACK after ACK w/ same ACK number, double RSTACK) */") \
_(SPT_rtr_RST, "/** Packet retransmission (PSHACK after ACK w/ same ACK number, double RSTACK) */") \
_(SPT_rtr_PSHACK, "/** Packet retransmission (PSHACK after ACK w/ same ACK number, double RSTACK) */") \
_(SPT_ooo_ACK, "/** Packet out of order (SYN after ACK, ACK number, ACK after RSTACK) */") \
_(SPT_ooo_RST, "/** Packet out of order (SYN after ACK, ACK number, ACK after RSTACK) */") \
_(SPT_ooo_PSHACK, "/** Packet out of order (SYN after ACK, ACK number, ACK after RSTACK) */") \
_(SPT_DUP_ACK, "/** Duplicated ACK */") \
_(SPT_DUP_PSHACK, "/** Duplicated PSHACK */") \
_(SPT_WEIRD, "/** Weird packet */") \
_(SPT_TS_INVALID, "/** Timestamp invalid */") \
_(SPT_BEYOND_SCOPE, "/** Out of scope */") 

#define lb_foreach_layout \
_(u8, n_as, 1, "Number of maximum ASs", SHM_N_BIN) \
_(msg_out_t, msg_out_cache, 1, "A cache for latest output message", 0) \
_(msg_out_t, msg_out_frames, 4, "A list of msg_out instance w/ increasing sequence id, written by VPP, read by SHM fetcher", 0) \
_(reservoir_as_t, res_as, 64, "An array of reservoir samples for each AS", 0) \
_(msg_in_t, msg_in_cache, 1, "A cache for latest input message, written/read by VPP", 0) \
_(msg_in_t, msg_in_frames, 4, "A list of msg_in instance w/ increasing sequence id, written by ML application, read by VPP", 0) 

