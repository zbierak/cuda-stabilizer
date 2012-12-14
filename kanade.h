#ifndef MZBIERSK_RIM_KANADE_H
#define MZBEIRSK_RIM_KANADE_H

const unsigned PYRAMID_SIZE = 5;
const unsigned ITERATION_COUNT = 5;
const unsigned WINDOW_SIZE_DIV_2 = 50;
const unsigned WINDOW_SIZE = 2*WINDOW_SIZE_DIV_2;

void kanadeInit();
void kanadeCleanup();
void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height);
void kanadeExecute(unsigned char* target, unsigned width, unsigned height);
void kanadePrintStats();

// for testing purposes only
void kanadeTestNextFrame(unsigned char* pixels, unsigned width, unsigned height);
void kanadeTestBuildPyramid(unsigned char* target, unsigned width, unsigned height);
void kanadeTestCompareBuildPyramid(unsigned char* target, unsigned width, unsigned height);
void kanadeTestGenerateG(unsigned char* target, unsigned width, unsigned height);
void kanadeTestGenerateB(unsigned char* target, unsigned width, unsigned height);

// utility functions
void kanade8to24(unsigned char* target, unsigned char* src, unsigned width, unsigned height);
void kanade24to8(unsigned char* target, unsigned char* src, unsigned width, unsigned height);

#endif