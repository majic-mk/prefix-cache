#define _GNU_SOURCE
#include <errno.h>
#include <linux/io_uring.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
int main(void) {
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    if (sizeof(p) != 120) {
        printf("{\"status\":\"UNSUPPORTED_ABI\",\"params_bytes\":%zu}\n",sizeof(p));
        return 3;
    }
    errno=0;
    long setup=syscall(__NR_io_uring_setup,2U,&p);
    int setup_errno=setup<0?errno:0;
    int closed=-1;
    if(setup>=0) closed=close((int)setup);
    errno=0;
    long enter=syscall(__NR_io_uring_enter,-1,0U,0U,0U,NULL,0UL);
    int enter_errno=enter<0?errno:0;
    errno=0;
    long reg=syscall(__NR_io_uring_register,-1,0U,NULL,0U);
    int reg_errno=reg<0?errno:0;
    printf("{\"params_bytes\":%zu,\"setup\":{\"syscall\":%ld,\"depth\":2,\"flags\":0,\"return\":%ld,\"errno\":%d,\"close_return\":%d},"
           "\"enter_invalid_fd\":{\"syscall\":%ld,\"fd\":-1,\"return\":%ld,\"errno\":%d},"
           "\"register_invalid_fd\":{\"syscall\":%ld,\"fd\":-1,\"return\":%ld,\"errno\":%d},"
           "\"io_submissions\":0,\"gpu_operations\":0}\n",
           sizeof(p),(long)__NR_io_uring_setup,setup,setup_errno,closed,
           (long)__NR_io_uring_enter,enter,enter_errno,(long)__NR_io_uring_register,reg,reg_errno);
    return setup<0?2:(closed==0?0:4);
}
