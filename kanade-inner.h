#ifndef MZBIERSK_RIM_KANADE_INNER
#define MZBIERSK_RIM_KANADE_INNER


// functions from CPU implementation
namespace kcpu {
void kanadeInit();
void kanadeCleanup();
void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height);
void kanadeBuildPyramidLevel(unsigned levelId, unsigned newWidth, unsigned newHeight);
void kanadeTranslate(unsigned char* target, float vx, float vy, unsigned width, unsigned height);
void kanadeCalculateG(unsigned pyrLvl, unsigned width, unsigned height, float& dxdx, float& dxdy, float& dydy);
void kanadeCalculateB(unsigned pyrLvl, float vx, float vy, unsigned width, unsigned height, float& bx, float& by);
void kanadePrepareForNextFrame(unsigned width[PYRAMID_SIZE], unsigned height[PYRAMID_SIZE]);
}

// functions from GPU implementation
#ifndef KANADE_NO_GPU
namespace kgpu {
void kanadeInit();
void kanadeCleanup();
void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height);
void kanadeBuildPyramidLevel(unsigned levelId, unsigned newWidth, unsigned newHeight);
void kanadeTranslate(unsigned char* target, float vx, float vy, unsigned width, unsigned height);
void kanadeCalculateG(unsigned pyrLvl, unsigned width, unsigned height, float& dxdx, float& dxdy, float& dydy);
void kanadeCalculateB(unsigned pyrLvl, float vx, float vy, unsigned width, unsigned height, float& bx, float& by);
void kanadePrepareForNextFrame(unsigned width[PYRAMID_SIZE], unsigned height[PYRAMID_SIZE]);
}
#endif

// functions used by common & test
unsigned getPyrWidth(unsigned lvl);
unsigned getPyrHeight(unsigned lvl);
void buildPyramid(unsigned width, unsigned height);

#endif