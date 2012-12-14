#ifndef MZBIERSK_RIM_KANADE_TEST
#define MZBIERSK_RIM_KANADE_TEST

// test functions for cpu implementation
namespace kcpu {
void getIoFrame8(unsigned char* target, unsigned lvl, unsigned width, unsigned height);
void getPrevFrame8(unsigned char* target, unsigned lvl, unsigned width, unsigned height);
void getDevDx(float* target, unsigned width, unsigned height);
void getDevDy(float* target, unsigned width, unsigned height);
void getDevDt(float* target, unsigned width, unsigned height);
}

// test functions for gpu implementation
#ifndef KANADE_NO_GPU
namespace kgpu {
void getIoFrame8(unsigned char* target, unsigned lvl, unsigned width, unsigned height);
void getPrevFrame8(unsigned char* target, unsigned lvl, unsigned width, unsigned height);
void getDevDx(float* target, unsigned width, unsigned height);
void getDevDy(float* target, unsigned width, unsigned height);
void getDevDt(float* target, unsigned width, unsigned height);
}
#endif

#endif