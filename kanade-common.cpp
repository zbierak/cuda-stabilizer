#include "kanade.h"
#include "kanade-inner.h"

#include <stdlib.h>

// select the execution version based on the state of preprocessor constants
#ifdef KANADE_NO_GPU
#define _current kcpu
#else
#define _current kgpu
#endif

unsigned pyrWidth[PYRAMID_SIZE] = {0};
unsigned pyrHeight[PYRAMID_SIZE] = {0};

unsigned getPyrWidth(unsigned lvl) { return pyrWidth[lvl]; }
unsigned getPyrHeight(unsigned lvl) { return pyrHeight[lvl]; }

void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height)
{
	_current::kanadeNextFrame(pixels, width, height);
}

unsigned char* kanadeExecute(unsigned width, unsigned height)
{
	// kod metody lukasa
	return NULL;
}

void buildPyramid(unsigned width, unsigned height)
{
	if (pyrWidth[0] == 0)
	{
		/* 
		 * prepare pyramid level sizes - if odd round down so as to prevent
		 * any problems with scaling the size
		 */
		for (unsigned w=width, h=height, i=0; i<PYRAMID_SIZE; i++)
		{
			pyrWidth[i] = w;
			pyrHeight[i] = h;

			w /= 2;
			h /= 2;

			if (w % 2 == 1) --w;
			if (h % 2 == 1) --h;
		}
	}

	for (unsigned i=1; i<PYRAMID_SIZE; i++)
		_current::kanadeBuildPyramidLevel(i, pyrWidth[i], pyrHeight[i]);
}


void kanadeInit()
{
	kcpu::kanadeInit();

	#ifndef KANADE_NO_GPU
	kgpu::kanadeInit();
	#endif
}

void kanadeCleanup()
{
	kcpu::kanadeCleanup();

	#ifndef KANADE_NO_GPU
	kgpu::kanadeCleanup();
	#endif
}

// undefine selectors
#undef _current