#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* struct */

long long 
get_mem (void)
{
    long long mem_free;
    FILE *fp;
    char line[50];

    fp = fopen("/proc/meminfo","r");
    fgets(line, sizeof(line), fp);
    fscanf(fp, "%*s %Ld", &mem_free);
    fclose(fp);
    
    return mem_free;
}