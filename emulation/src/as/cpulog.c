#include "cpulog.h"

log_cpu_t cpu_main;

int
log_cpu_t_stat_scan(log_cpu_t * self)
{
        FILE * file;
        char * ret;
        char buffer[256];

        /* try open file */
        file = fopen("/proc/stat", "r");
    	if (file == NULL) 
        {
        	perror("Could not open stat file");
        	return 1;
    	}
        
        /* get buffer */
        ret = fgets(buffer, sizeof(buffer) - 1, file);
	if (ret == NULL)
	{
        	perror("Could not read stat file");
        	fclose(file);
        	return 1;
    	}
        
        /* close file */
        fclose(file);
        
        sscanf(buffer,
           	"cpu  %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu",
           	&self->user, &self->nice, &self->system, &self->idle, &self->iowait, &self->irq, &self->softirq, &self->steal, &self->guest, &self->guestnice);
        return 0;
}

static inline long long unsigned int
log_cpu_t_cal_total (log_cpu_t * self)
{
        return self->user + self->nice + self->system + self->idle + self->iowait + self->irq + self->softirq + self->steal;
}

static inline long long unsigned int
log_cpu_t_cal_idle (log_cpu_t * self)
{
        return self->idle + self->iowait;
}

static inline double
log_cpu_t_cal_load (log_cpu_t * self)
{
        return ((self->total - self->prev_total) - (self->idle_sum - self->prev_idle_sum)) / (self->total - self->prev_total + 1.e-10);
}

void
cpu_init (log_cpu_t * lc)
{
        lc->f.cal_total = log_cpu_t_cal_total;
        lc->f.cal_idle = log_cpu_t_cal_idle;
        lc->f.get_stat = log_cpu_t_stat_scan;
        lc->f.cal_load = log_cpu_t_cal_load;
        /* get initial value */
	lc->f.get_stat(lc);
        lc->prev_total = lc->f.cal_total(lc);
        lc->prev_idle_sum = lc->f.cal_idle(lc);
}

double
get_update_cpu (log_cpu_t * lc)
{
        lc->f.get_stat(lc);
        lc->total = lc->f.cal_total(lc);
        lc->idle_sum = lc->f.cal_idle(lc);
        lc->loadavg = lc->f.cal_load(lc);
        lc->prev_total = lc->total;
        lc->prev_idle_sum = lc->idle_sum;
        return lc->loadavg;
}
