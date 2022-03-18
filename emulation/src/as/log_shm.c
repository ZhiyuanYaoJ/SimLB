#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <stdint.h>
#include "cpulog.h"
#include "memlog.h"
#include "apache_agent.h"

#define SHM_SZ 1048576

typedef struct {
        double cpu;
        long long mem;
        int apache;
        int id;
} __attribute__((packed)) msg;

int main(int argc, char *argv[])
{
        int i, fd, id;
        char* mem;
        long double dt = 0.10000;
        int dt_sleep = (int)(dt * 1000000);
        if ((fd = shm_open("shm_vpp", O_RDWR | O_CREAT, 0777)) < 0) {
                perror("shm_open");
                return -1;
        }
        if (ftruncate(fd, SHM_SZ) < 0) {
                perror("ftruncate");
                return -1;
        }
        if ((mem = mmap(0, SHM_SZ, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED) {
                perror("mmap");
                return -1;
        }

        log_cpu_t * lc = &cpu_main;

        // get currrent server id
        id = atoi(argv[1]);
        
        // Let's put some message at address [mem+42->mem+110), and retrieve one from address [mem+110->mem+178)
  
        struct msg_py {
                uint32_t int1;
                uint32_t int2;
                char a_str[64];
        } __attribute__((packed));

        msg *msg_out = (msg*)(mem + 42);
        struct msg_py *msg_in = (struct msg_py*)(mem + 110);

        msg_out->id = id;
        cpu_init(lc);

        for(i=0; ; i++) {
                msg_out->cpu = get_update_cpu(lc);
                msg_out->mem = get_mem();
                msg_out->apache = get_apache_agent();
                // snprintf(msg_out->a_str, sizeof(msg_out->a_str), "That's the %d-th time something happens\n", i);
                // printf("%d %d %s", msg_in->int1, msg_in->int2, msg_in->a_str);
                usleep(dt_sleep);
        }

        return 0;
}
