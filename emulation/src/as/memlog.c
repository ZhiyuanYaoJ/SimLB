#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


int main(void)
{
    long long mem_free;
    FILE *fp;
    char line[50];

    long double dt = 0.10000;
    int dt_sleep = (int)(dt * 1000000);

    for(;;)
    {
        fp = fopen("/proc/meminfo","r");
        fgets(line, sizeof line, fp);
        fscanf(fp, "%*s %Ld", &mem_free);
        fclose(fp);
        printf("%Ld\n",mem_free);
        usleep(dt_sleep);
    }

    return(0);
}
