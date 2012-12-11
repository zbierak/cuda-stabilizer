#ifndef MZBIERSK_RIM_KANADE_H
#define MZBEIRSK_RIM_KANADE_H

void kanadeInit();
void kanadeCleanup();
void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height);
unsigned char* kanadeExecute(unsigned width, unsigned height);

// the following to be used by kanade-common only, assumed current frame is already loaded

unsigned char* kanadeTranslate(unsigned width, unsigned height);
void calculateG(unsigned width, unsigned height, float& dxdx, float& dxdy, float& dydy);

#endif