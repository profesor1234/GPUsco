/*
 * Scorbits (SCO) GPU Miner v9 — Optimized for GTX 1060 (Pascal sm_61)
 *
 * Optimizations:
 *   - Fast-Nonce ASCII: Optimized integer-to-string conversion in kernel.
 *   - Unrolled SHA-256: Manual round unrolling for Pascal architecture.
 *   - SMEM optimization: Improved shared memory layout.
 *   - Grid Tuning: Scaled for 10 SMs (GTX 1060).
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
  #include <winhttp.h>
  #pragma comment(lib, "winhttp.lib")
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <errno.h>
  #include <sys/time.h>
  #include <curl/curl.h>
  #include <unistd.h>
  typedef int sock_t;
  #define INVALID_SOCK (-1)
  #define CLOSE_SOCK close
#endif

/* ── Platform sleep ─────────────────────────────────────────────── */
static void msleep(int ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    usleep((useconds_t)ms * 1000);
#endif
}

#ifndef _WIN32
/* ── Linux HTTP/HTTPS via libcurl ───────────────────────────────── */
typedef struct { char* buf; int len; int cap; } CurlBuf;
static size_t curl_cb(void* p, size_t sz, size_t nm, void* ud) {
    CurlBuf* b=(CurlBuf*)ud; int n=(int)(sz*nm);
    if(b->len+n>=b->cap-1) return 0;
    memcpy(b->buf+b->len,p,n); b->len+=n; b->buf[b->len]='\0'; return n;
}
static int http_request(const char* method, const char* host, int port,
                        const char* path, const char* body,
                        char* out, int outsz, int* status_code) {
    CURL* curl=curl_easy_init(); if(!curl){if(status_code)*status_code=0;return -1;}
    char url[512];
    const char* scheme=(port==443)?"https":"http";
    if(port==80||port==443) snprintf(url,sizeof(url),"%s://%s%s",scheme,host,path);
    else snprintf(url,sizeof(url),"%s://%s:%d%s",scheme,host,port,path);
    CurlBuf b={out,0,outsz}; out[0]='\0';
    curl_easy_setopt(curl,CURLOPT_URL,url);
    curl_easy_setopt(curl,CURLOPT_WRITEFUNCTION,curl_cb);
    curl_easy_setopt(curl,CURLOPT_WRITEDATA,&b);
    curl_easy_setopt(curl,CURLOPT_TIMEOUT,30L);
    curl_easy_setopt(curl,CURLOPT_SSL_VERIFYPEER,0L);
    curl_easy_setopt(curl,CURLOPT_SSL_VERIFYHOST,0L);
    struct curl_slist* hdrs=NULL;
    if(body&&strlen(body)>0){
        hdrs=curl_slist_append(hdrs,"Content-Type: application/json");
        curl_easy_setopt(curl,CURLOPT_HTTPHEADER,hdrs);
        curl_easy_setopt(curl,CURLOPT_POSTFIELDS,body);
    }
    CURLcode rc=curl_easy_perform(curl);
    long http_code=0; curl_easy_getinfo(curl,CURLINFO_RESPONSE_CODE,&http_code);
    if(status_code)*status_code=(int)http_code;
    if(hdrs) curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);
    return (rc==CURLE_OK)?b.len:-1;
}
#endif


/* --- SHA-256 Macros --- */
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z)   (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z)  (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x)      (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define EP1(x)      (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define SIG0(x)     (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x)     (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))

__constant__ uint32_t K_GPU[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

static const uint32_t K_CPU[64]={
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

/* --- SHA-256 Optimized Compression --- */
__device__ __forceinline__ void sha256_compress_optimized(uint32_t state[8], const uint32_t w16[16]) {
    uint32_t w[64];
    #pragma unroll 16
    for(int i=0;i<16;i++) w[i]=w16[i];
    #pragma unroll 48
    for(int i=16;i<64;i++) w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
    
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    
    #pragma unroll 64
    for(int i=0;i<64;i++){
        uint32_t t1=h+EP1(e)+CH(e,f,g)+K_GPU[i]+w[i];
        uint32_t t2=EP0(a)+MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

__device__ __forceinline__ int fast_itoa(char* buf, long long v) {
    if(v == 0) { buf[0] = '0'; return 1; }
    char tmp[20]; int i = 0;
    while(v > 0) {
        tmp[i++] = '0' + (int)(v % 10);
        v /= 10;
    }
    for(int j = 0; j < i; j++) buf[j] = tmp[i - 1 - j];
    return i;
}

__global__ __launch_bounds__(256, 4) void mine_kernel_optimized(
    const uint32_t* __restrict__ d_midstate, const char* __restrict__ d_tail, int tail_len,
    const char* __restrict__ d_suffix, int slen, int prefix_total_len,
    long long base_nonce, int difficulty,
    long long* found_nonce, long long* found_ts, long long current_ts, char* found_hash_hex)
{
    extern __shared__ char smem[];
    char* s_tail = smem; 
    char* s_suffix = smem + tail_len;
    
    int tid = threadIdx.x;
    for(int i = tid; i < tail_len; i += blockDim.x) s_tail[i] = d_tail[i];
    for(int i = tid; i < slen; i += blockDim.x) s_suffix[i] = d_suffix[i];
    __syncthreads();

    long long nonce = base_nonce + (long long)(blockIdx.x * blockDim.x + tid);

    uint8_t data[128]; 
    int pos = 0;
    for(int i = 0; i < tail_len; i++) data[pos++] = (uint8_t)s_tail[i];
    
    char ns[20]; 
    int nl = fast_itoa(ns, nonce);
    for(int i = 0; i < nl; i++) data[pos++] = (uint8_t)ns[i];
    for(int i = 0; i < slen; i++) data[pos++] = (uint8_t)s_suffix[i];

    uint32_t total_len = (uint32_t)(prefix_total_len + nl + slen);
    data[pos] = 0x80;
    int padded_len = ((pos + 1 + 8) + 63) & ~63;
    for(int i = pos + 1; i < padded_len; i++) data[i] = 0;
    
    uint64_t bits = (uint64_t)total_len * 8ULL;
    data[padded_len - 1] = (uint8_t)(bits);       data[padded_len - 2] = (uint8_t)(bits >> 8);
    data[padded_len - 3] = (uint8_t)(bits >> 16);  data[padded_len - 4] = (uint8_t)(bits >> 24);
    data[padded_len - 5] = (uint8_t)(bits >> 32);  data[padded_len - 6] = (uint8_t)(bits >> 40);
    data[padded_len - 7] = (uint8_t)(bits >> 48);  data[padded_len - 8] = (uint8_t)(bits >> 56);

    uint32_t state[8];
    state[0] = d_midstate[0]; state[1] = d_midstate[1]; state[2] = d_midstate[2]; state[3] = d_midstate[3];
    state[4] = d_midstate[4]; state[5] = d_midstate[5]; state[6] = d_midstate[6]; state[7] = d_midstate[7];

    int rem_blocks = padded_len / 64;
    for(int b = 0; b < rem_blocks; b++) {
        uint32_t w16[16]; uint8_t* bp = data + b * 64;
        #pragma unroll 16
        for(int i = 0; i < 16; i++) 
            w16[i] = ((uint32_t)bp[i * 4] << 24) | ((uint32_t)bp[i * 4 + 1] << 16) | ((uint32_t)bp[i * 4 + 2] << 8) | bp[i * 4 + 3];
        sha256_compress_optimized(state, w16);
    }

    if(difficulty >= 4 && (state[0] >> 16) != 0) return;
    if(difficulty >= 2 && (state[0] >> 24) != 0) return;

    uint8_t hash[32];
    #pragma unroll 8
    for(int i = 0; i < 8; i++) {
        uint32_t v = state[i];
        hash[i * 4] = (v >> 24) & 0xFF; hash[i * 4 + 1] = (v >> 16) & 0xFF; hash[i * 4 + 2] = (v >> 8) & 0xFF; hash[i * 4 + 3] = v & 0xFF;
    }

    int full = difficulty / 2;
    bool ok = true;
    for(int i = 0; i < full && ok; i++) if(hash[i] != 0x00u) ok = false;
    if(ok && (difficulty & 1)) if((hash[full] >> 4) != 0) ok = false;

    if(ok) {
        unsigned long long prev = atomicCAS((unsigned long long*)found_nonce, (unsigned long long)(-1LL), (unsigned long long)nonce);
        if(prev == (unsigned long long)(-1LL)) {
            *found_ts = current_ts;
            const char hx[] = "0123456789abcdef";
            for(int i = 0; i < 32; i++) {
                found_hash_hex[i * 2] = hx[hash[i] >> 4];
                found_hash_hex[i * 2 + 1] = hx[hash[i] & 0xF];
            }
            found_hash_hex[64] = '\0';
        }
    }
}

/* --- Host Logic --- */
typedef struct { uint32_t h[8]; char tail[256]; int tail_len; int total_prefix_len; } Midstate;

static void sha256_block_cpu(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for(int i=0;i<16;i++) w[i]=((uint32_t)block[i*4]<<24)|((uint32_t)block[i*4+1]<<16)|((uint32_t)block[i*4+2]<<8)|block[i*4+3];
    for(int i=16;i<64;i++) w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for(int i=0;i<64;i++){uint32_t t1=h+EP1(e)+CH(e,f,g)+K_CPU[i]+w[i];uint32_t t2=EP0(a)+MAJ(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

static void compute_midstate(const char* prefix, int plen, Midstate* ms) {
    ms->h[0]=0x6a09e667u;ms->h[1]=0xbb67ae85u;ms->h[2]=0x3c6ef372u;ms->h[3]=0xa54ff53au;
    ms->h[4]=0x510e527fu;ms->h[5]=0x9b05688cu;ms->h[6]=0x1f83d9abu;ms->h[7]=0x5be0cd19u;
    ms->total_prefix_len=plen;
    int full_blocks=plen/64;
    for(int b=0;b<full_blocks;b++) sha256_block_cpu(ms->h,(const uint8_t*)prefix+b*64);
    ms->tail_len=plen-full_blocks*64;
    if(ms->tail_len>0) memcpy(ms->tail,prefix+full_blocks*64,ms->tail_len);
}

#ifdef _WIN32
static int http_request(const char* method, const char* host, int port,
                        const char* path, const char* body,
                        char* out, int outsz, int* status_code) {
    if(status_code) *status_code=0; out[0]='\0';
    wchar_t whost[256]; MultiByteToWideChar(CP_ACP,0,host,-1,whost,256);
    HINTERNET hSession=WinHttpOpen(L"SCO-GPU/9",WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,0,0,0);
    HINTERNET hConnect=WinHttpConnect(hSession,whost,(INTERNET_PORT)port,0);
    wchar_t wpath[512]; MultiByteToWideChar(CP_ACP,0,path,-1,wpath,512);
    wchar_t* wmeth=(strcmp(method,"POST")==0)?L"POST":L"GET";
    HINTERNET hReq=WinHttpOpenRequest(hConnect,wmeth,wpath,0,0,0, (port==443?WINHTTP_FLAG_SECURE:0));
    if(port==443){DWORD opt=SECURITY_FLAG_IGNORE_UNKNOWN_CA|SECURITY_FLAG_IGNORE_CERT_DATE_INVALID|SECURITY_FLAG_IGNORE_CERT_CN_INVALID;
        WinHttpSetOption(hReq,WINHTTP_OPTION_SECURITY_FLAGS,&opt,sizeof(opt));}
    BOOL sent;
    if(body&&strlen(body)>0){
        sent=WinHttpSendRequest(hReq,L"Content-Type: application/json\r\n",(DWORD)-1,(LPVOID)body,(DWORD)strlen(body),(DWORD)strlen(body),0);
    } else {
        sent=WinHttpSendRequest(hReq,WINHTTP_NO_ADDITIONAL_HEADERS,0,WINHTTP_NO_REQUEST_DATA,0,0,0);
    }
    if(!sent){WinHttpCloseHandle(hReq);WinHttpCloseHandle(hConnect);WinHttpCloseHandle(hSession);return -1;}
    WinHttpReceiveResponse(hReq,0);
    DWORD sc=0,scl=sizeof(sc); WinHttpQueryHeaders(hReq,WINHTTP_QUERY_STATUS_CODE|WINHTTP_QUERY_FLAG_NUMBER,0,&sc,&scl,0);
    if(status_code)*status_code=(int)sc;
    DWORD avail=0,read=0; int n=0;
    while(WinHttpQueryDataAvailable(hReq,&avail)&&avail>0&&n<outsz-1){
        DWORD tr=avail<(DWORD)(outsz-1-n)?avail:(DWORD)(outsz-1-n);
        WinHttpReadData(hReq,out+n,tr,&read); n+=read;
    }
    out[n]=0; WinHttpCloseHandle(hReq); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); return n;
}
#endif

static void json_unescape(char* s) {
    char* r=s, *w=s;
    while(*r) {
        if(r[0]=='\\' && r[1]=='u' && r[2]=='0' && r[3]=='0') {
            unsigned int cp=0; sscanf(r+2,"u%04x",&cp);
            if(cp<128) { *w++=(char)cp; r+=6; continue; }
        }
        if(r[0]=='\\' && r[1]=='n'){*w++='\n';r+=2;continue;}
        if(r[0]=='\\' && r[1]=='"'){*w++='"';r+=2;continue;}
        if(r[0]=='\\' && r[1]=='\\'){*w++='\\';r+=2;continue;}
        *w++=*r++;
    }
    *w='\0';
}
static int jstr(const char* js,const char* key,char* out,int sz){
    char pat[128];sprintf(pat,"\"%s\":",key);const char* p=strstr(js,pat);if(!p)return 0;
    p+=strlen(pat);while(*p==' ')p++;int i=0;
    if(*p=='"'){p++;while(*p&&*p!='"'&&i<sz-1)out[i++]=*p++;}
    else{while(*p&&((*p>='0'&&*p<='9')||*p=='-'||*p=='.')&&i<sz-1)out[i++]=*p++;}
    out[i]='\0';json_unescape(out);return i>0;
}
static int jbool(const char* js,const char* key){
    char pat[128];sprintf(pat,"\"%s\":",key);const char* p=strstr(js,pat);if(!p)return -1;
    p+=strlen(pat);while(*p==' ')p++;
    if(strncmp(p,"true",4)==0)return 1;if(strncmp(p,"false",5)==0)return 0;return -1;
}

typedef struct{int block_index;char previous_hash[256];int difficulty;int reward;long long timestamp;long long last_timestamp;char transactions[2048];}WorkTemplate;

static int fetch_work(const char* host,int port,const char* address,WorkTemplate* work){
    static char resp[8192]; memset(resp,0,sizeof(resp)); int status=0;
    char path[256]; snprintf(path,sizeof(path),"/mining/work?address=%s",address);
    http_request("GET",host,port,path,NULL,resp,sizeof(resp),&status);
    if(status!=200||!resp[0]) return 0; char val[256];
    if(!jstr(resp,"block_index",val,sizeof(val))) return 0; work->block_index=atoi(val);
    jstr(resp,"previous_hash",work->previous_hash,sizeof(work->previous_hash));
    if(!jstr(resp,"difficulty",val,sizeof(val))) return 0; work->difficulty=atoi(val);
    jstr(resp,"reward",val,sizeof(val)); work->reward=atoi(val);
    jstr(resp,"timestamp",val,sizeof(val)); work->timestamp=atoll(val);
    jstr(resp,"last_timestamp",val,sizeof(val)); work->last_timestamp=atoll(val);
    const char* ta=strstr(resp,"\"transactions\":");
    if(ta){const char* br=strchr(ta,'[');if(br){br++;char items[2048]={0};int ilen=0;
        while(*br&&*br!=']'){while(*br==' '||*br==','||*br=='\n'||*br=='\r')br++;
        if(*br=='"'){br++;while(*br&&*br!='"'&&ilen<(int)sizeof(items)-2)items[ilen++]=*br++;
            if(*br=='"')br++;while(*br==' ')br++;if(*br==',')items[ilen++]=';';}}
        items[ilen]='\0'; strncpy(work->transactions,items,sizeof(work->transactions)-1);}}
    if(!work->transactions[0])strcpy(work->transactions,"empty-block"); return 1;
}

typedef struct{int success;int block_index;int reward;char hash[128];char error[256];int http_status;}SubmitResult;

static void submit_block(const char* host,int port,const WorkTemplate* work,
    long long nonce,long long ts,const char* hash_hex,const char* address,SubmitResult* result){
    char tx_json[512],tx_copy[512]; strncpy(tx_copy,work->transactions,sizeof(tx_copy)-1); tx_copy[511]=0;
    tx_json[0]='['; int jpos=1; char* tok=strtok(tx_copy,";"); int first=1;
    while(tok){if(!first)tx_json[jpos++]=','; tx_json[jpos++]='"'; while(*tok&&jpos<(int)sizeof(tx_json)-4)tx_json[jpos++]=*tok++; tx_json[jpos++]='"'; first=0; tok=strtok(NULL,";");}
    tx_json[jpos++]=']'; tx_json[jpos]='\0';
    char body[2048];
    sprintf(body,"{\"block_index\":%d,\"nonce\":%lld,\"hash\":\"%s\",\"miner_address\":\"%s\",\"timestamp\":%lld,\"transactions\":%s}",
        work->block_index,(long long)nonce,hash_hex,address,(long long)ts,tx_json);
    static char resp[2048]; memset(resp,0,sizeof(resp)); int status=0;
    http_request("POST",host,port,"/mining/submit",body,resp,sizeof(resp),&status);
    result->http_status=status; result->success=0; result->error[0]='\0';
    if(jbool(resp,"success")==1){result->success=1; char val[64];
        jstr(resp,"block_index",val,sizeof(val)); result->block_index=atoi(val);
        jstr(resp,"reward",val,sizeof(val)); result->reward=atoi(val);
        jstr(resp,"hash",result->hash,sizeof(result->hash));}
    else{jstr(resp,"error",result->error,sizeof(result->error)); if(!result->error[0])strncpy(result->error,resp,sizeof(result->error)-1);}
}

static double get_time() {
#ifdef _WIN32
    LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart/(double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
#endif
}

int main(int argc, char** argv) {
#ifndef _WIN32
    curl_global_init(CURL_GLOBAL_ALL);
#endif
    char address[128]="", node_host[128]="scorbits.com";
    int node_port=443;
    for(int i=1;i<argc;i++){
        if((strcmp(argv[i],"--address")==0||strcmp(argv[i],"-a")==0)&&i+1<argc) strncpy(address,argv[++i],sizeof(address)-1);
        else if((strcmp(argv[i],"--node")==0||strcmp(argv[i],"-n")==0)&&i+1<argc){
            i++; char* url=argv[i];
            if(strncmp(url,"https://",8)==0){url+=8;node_port=443;}
            else if(strncmp(url,"http://",7)==0){url+=7;node_port=80;}
            char* col=strrchr(url,':'); if(col){int hl=(int)(col-url); strncpy(node_host,url,hl); node_host[hl]='\0'; node_port=atoi(col+1);}
            else strncpy(node_host,url,sizeof(node_host)-1);
        }
        else if(strncmp(argv[i],"SCO",3)==0) strncpy(address,argv[i],sizeof(address)-1);
    }
    if(!address[0]){printf("SCO address: ");scanf("%127s",address);}

    printf("\n=== Scorbits GPU Miner v9 — Windows / Linux ===\n");
    printf("Node: %s:%d | Address: %s\n\n",node_host,node_port,address);

    int dev_count=0; cudaGetDeviceCount(&dev_count);
    if(dev_count==0){printf("[ERROR] No CUDA GPU!\n");return 1;}
    int dev=0; cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
    printf("[GPU 0] %s | %d SMs | CUDA %d.%d\n",prop.name,prop.multiProcessorCount,prop.major,prop.minor);

    int tpb=256, bpg=prop.multiProcessorCount*128; // 128 blocks per SM
    long long batch=(long long)tpb*bpg;

    uint32_t *d_midstate; char *d_tail,*d_suffix,*d_found_hash; long long *d_found_nonce,*d_found_ts_dev;
    cudaMalloc(&d_midstate,8*sizeof(uint32_t)); cudaMalloc(&d_tail,2048); cudaMalloc(&d_suffix,256);
    cudaMalloc(&d_found_nonce,sizeof(long long)); cudaMalloc(&d_found_ts_dev,sizeof(long long)); cudaMalloc(&d_found_hash,65);

    WorkTemplate work; long long last_accepted_ts=0, total_blocks=0; double session_start=get_time();
    int last_mined_block=-1;

    for(;;){
        printf("[Work] Fetching...\n"); memset(&work,0,sizeof(work));
        if(!fetch_work(node_host,node_port,address,&work)){printf("[Work] Failed — 5s\n");msleep(5000);continue;}
        printf("[Work] #%d | diff=%d | reward=%d\n",work.block_index,work.difficulty,work.reward);

        if(work.block_index == last_mined_block + 1){
            printf("[Skip] Consecutive block #%d — waiting...\n", work.block_index);
            while(work.block_index == last_mined_block + 1){ msleep(3000); fetch_work(node_host,node_port,address,&work); }
        }

        long long server_now = work.timestamp, wall_start = (long long)time(NULL), min_ts = work.last_timestamp + 123;
        long long ts = (server_now > min_ts) ? server_now : min_ts;
        char prefix[2048]; int plen=snprintf(prefix,sizeof(prefix),"%d%lld%s%s",work.block_index,(long long)ts,work.transactions,work.previous_hash);
        Midstate ms; compute_midstate(prefix,plen,&ms);

        int slen=(int)strlen(address);
        cudaMemcpy(d_midstate,ms.h,8*sizeof(uint32_t),cudaMemcpyHostToDevice);
        cudaMemcpy(d_tail,ms.tail,ms.tail_len,cudaMemcpyHostToDevice);
        cudaMemcpy(d_suffix,address,slen,cudaMemcpyHostToDevice);

        long long h_nonce=-1LL, base=0; double t0=get_time(), tr=t0, poll=t0;
        for(;;){
            long long new_ts = server_now + ((long long)time(NULL) - wall_start);
            if(new_ts < min_ts) new_ts = min_ts;
            if(new_ts!=ts){
                ts=new_ts; plen=snprintf(prefix,sizeof(prefix),"%d%lld%s%s",work.block_index,ts,work.transactions,work.previous_hash);
                compute_midstate(prefix,plen,&ms);
                cudaMemcpy(d_midstate,ms.h,8*sizeof(uint32_t),cudaMemcpyHostToDevice);
                cudaMemcpy(d_tail,ms.tail,ms.tail_len,cudaMemcpyHostToDevice);
            }

            cudaMemcpy(d_found_nonce,&h_nonce,sizeof(long long),cudaMemcpyHostToDevice);
            mine_kernel_optimized<<<bpg,tpb,ms.tail_len+slen>>>(d_midstate,d_tail,ms.tail_len,d_suffix,slen,ms.total_prefix_len,
                base,work.difficulty,d_found_nonce,d_found_ts_dev,ts,d_found_hash);
            cudaDeviceSynchronize();
            base+=batch;

            cudaMemcpy(&h_nonce,d_found_nonce,sizeof(long long),cudaMemcpyDeviceToHost);
            if(h_nonce!=-1LL) break;

            double now2=get_time();
            if(now2-tr>=3.0){ printf("[GPU] #%d | %.2f MH/s | %lld H\n",work.block_index,base/(now2-t0)/1e6,base); tr=now2; }
            if(now2-poll>=3.0){
                poll=now2; WorkTemplate fresh;
                if(fetch_work(node_host,node_port,address,&fresh)&&(fresh.block_index!=work.block_index||fresh.difficulty!=work.difficulty)){break;}
            }
        }

        if(h_nonce!=-1LL){
            char h_hash[65]; cudaMemcpy(h_hash,d_found_hash,64,cudaMemcpyDeviceToHost); h_hash[64]=0;
            long long h_found_ts; cudaMemcpy(&h_found_ts,d_found_ts_dev,sizeof(long long),cudaMemcpyDeviceToHost);
            printf("[Found!] #%d nonce=%lld ts=%lld\n",work.block_index,h_nonce,h_found_ts);
            SubmitResult sr; submit_block(node_host,node_port,&work,h_nonce,h_found_ts,h_hash,address,&sr);
            if(sr.success){ total_blocks++; last_mined_block=work.block_index; printf("[Accepted] #%d Total:%lld\n",sr.block_index,total_blocks); }
            else { printf("[Rejected] %s\n",sr.error); }
        }
    }
    return 0;
}
