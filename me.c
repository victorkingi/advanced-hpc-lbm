#include <proccpuinfo.h>
#include <stdio.h>

int main() {
        proccpuinfo *info = proccpuinfo_read();

        if (!info)
                return 1;

        printf("architecture\t\t: %s\n",info->architecture);
        printf("hardware_platform\t: %s\n",info->hardware_platform);
        printf("frequency\t\t: %lf MHz\n", info->frequency);
        printf("cache\t\t\t: %d KB\n", info->cache);
        printf("cpus\t\t\t: %d processor%c\n",info->cpus, (info->cpus == 1 ? ' ' : 's'));
        printf("bogomips\t\t: %lf\n",info->bogomips);

        proccpuinfo_free(info);
        return 0;
}
