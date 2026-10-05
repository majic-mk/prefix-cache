#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
static int phase(aio_context_t ctx,int fd,void *buf,int write_op){
    struct iocb cb[2]; struct iocb *req[2]; struct io_event ev[2];
    memset(cb,0,sizeof(cb)); memset(ev,0,sizeof(ev));
    for(int i=0;i<2;i++){
        cb[i].aio_data=(unsigned long long)(i+1);
        cb[i].aio_lio_opcode=write_op?IOCB_CMD_PWRITE:IOCB_CMD_PREAD;
        cb[i].aio_fildes=fd;
        cb[i].aio_buf=(unsigned long long)(unsigned long)((char*)buf+i*4096);
        cb[i].aio_nbytes=4096; cb[i].aio_offset=i*4096; req[i]=&cb[i];
    }
    errno=0; long n=syscall(__NR_io_submit,ctx,2L,req); int err=n<0?errno:0;
    printf("{\"operation\":\"%s_submit\",\"submitted\":%ld,\"errno\":%d}\n",write_op?"write":"read",n,err);
    if(n!=2) return 2;
    struct timespec timeout={3,0};
    errno=0; long got=syscall(__NR_io_getevents,ctx,2L,2L,ev,&timeout); err=got<0?errno:0;
    printf("{\"operation\":\"%s_getevents\",\"events\":%ld,\"errno\":%d}\n",write_op?"write":"read",got,err);
    if(got!=2) return 3;
    unsigned seen=0;
    for(int i=0;i<2;i++){
        printf("{\"id\":%llu,\"bytes\":%lld,\"res2\":%lld}\n",(unsigned long long)ev[i].data,(long long)ev[i].res,(long long)ev[i].res2);
        if(ev[i].data<1 || ev[i].data>2 || ev[i].res!=4096 || ev[i].res2!=0) return 4;
        unsigned bit=1U<<(ev[i].data-1);
        if(seen&bit) return 5;
        seen|=bit;
    }
    return seen==3?0:6;
}
int main(int argc,char **argv){
    if(argc!=2) return 64;
    setvbuf(stdout,NULL,_IONBF,0);
    aio_context_t ctx=0;
    errno=0; long r=syscall(__NR_io_setup,2U,&ctx); int err=r<0?errno:0;
    printf("{\"operation\":\"io_setup\",\"depth\":2,\"return\":%ld,\"errno\":%d}\n",r,err);
    if(r<0) return 2;
    int fd=-1,rc=3;
    void *w=NULL,*b=NULL;
    if(posix_memalign(&w,4096,8192)!=0 || posix_memalign(&b,4096,8192)!=0) goto done;
    for(int i=0;i<8192;i++) ((unsigned char*)w)[i]=(unsigned char)((i*17+91)%251);
    memset(b,0,8192);
    errno=0; fd=open(argv[1],O_CREAT|O_EXCL|O_RDWR|O_DIRECT|O_CLOEXEC,0600);
    printf("{\"operation\":\"open_O_DIRECT_exclusive\",\"success\":%s,\"errno\":%d}\n",fd>=0?"true":"false",fd<0?errno:0);
    if(fd<0) goto done;
    if(phase(ctx,fd,w,1)!=0) goto done;
    if(fsync(fd)!=0) goto done;
    if(phase(ctx,fd,b,0)!=0) goto done;
    rc=memcmp(w,b,8192)==0?0:4;
    printf("{\"operation\":\"compare\",\"bytes\":8192,\"identical\":%s}\n",rc==0?"true":"false");
done:
    errno=0; long destroyed=syscall(__NR_io_destroy,ctx); int destroy_errno=destroyed<0?errno:0;
    printf("{\"operation\":\"io_destroy\",\"return\":%ld,\"errno\":%d}\n",destroyed,destroy_errno);
    if(destroyed!=0) rc=5;
    if(fd>=0 && close(fd)!=0) rc=6;
    free(w); free(b);
    printf("{\"status\":\"%s\",\"gpu_operations\":0,\"production_integration\":false}\n",rc==0?"CPU_LINUX_AIO_ODIRECT_ROUNDTRIP_PASS":"FAIL");
    return rc;
}
