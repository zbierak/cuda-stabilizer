#include "kanade.h"
#include <stdlib.h>

namespace kcpu {

unsigned char* ioFrame24 = NULL;									// pamiec pod ramke 24b - uzywana na wejsciu i wyjsciu
unsigned char* ioFrame8[PYRAMID_SIZE] = { NULL };					// pamiec pod piramide ramke 8b  - uzywana na wejsciu jako tymczasowa dla tekstury
unsigned char* prevFrame8[PYRAMID_SIZE] = { NULL };					// poprzednia piramida ramek w formacie osmiobitowym - do wyliczania dt


void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height)
{
	ioFrame24 = pixels;

	for (unsigned i=0; i<PYRAMID_SIZE; ++i)
	{
		if (ioFrame8[i] != NULL)
		{
			if (prevFrame8[i] != NULL)
				free(prevFrame8[i]);

			prevFrame8[i] = ioFrame8[i];
			ioFrame8[i] = (unsigned char*)malloc(width * height * sizeof(unsigned char));
		}
	}
}

void kanadeInit() { }

void kanadeCleanup()
{
	for (unsigned i=0; i<PYRAMID_SIZE; ++i)
	{
		if (ioFrame8[i] != NULL)
			free(ioFrame8[i]);
		if (prevFrame8[i] != NULL)
			free(prevFrame8[i]);
	}
}

void kanadeBuildPyramidLevel(unsigned levelId, unsigned newWidth, unsigned newHeight) {}

unsigned char* kanadeTranslate(unsigned width, unsigned height) { return NULL; }

void kanadeCalculateG(unsigned pyrLvl, unsigned width, unsigned height, float& dxdx, float& dxdy, float& dydy) {}

void kanadeCalculateB(unsigned pyrLvl, float vx, float vy, unsigned width, unsigned height, float& b) {}

// for testing purposes
void getIoFrame8(unsigned char* target, unsigned lvl, unsigned width, unsigned height) {}

}