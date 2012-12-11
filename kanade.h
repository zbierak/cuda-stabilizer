#ifndef MZBIERSK_RIM_KANADE_H
#define MZBEIRSK_RIM_KANADE_H

const unsigned PYRAMID_SIZE = 5;

void kanadeInit();
void kanadeCleanup();
void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height);
unsigned char* kanadeExecute(unsigned width, unsigned height);

// for testing purposes only

void kanadeTestBuildPyramid(unsigned char* target, unsigned width, unsigned height);
void kanade8to24(unsigned char* target, unsigned char* src, unsigned width, unsigned height);

#endif