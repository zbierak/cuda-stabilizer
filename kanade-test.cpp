#include "kanade.h"
#include "kanade-inner.h"
#include "kanade-test.h"

#include <stdlib.h>

#ifdef KANADE_NO_GPU
#define _current kcpu
#else
#define _current kgpu
#endif
void kanadeTestBuildPyramid(unsigned char* target, unsigned width, unsigned height)
{
	buildPyramid(width, height);
	_current::getIoFrame8(target, 0, width, height);
	
	unsigned shx = 0;
	unsigned shy = 0;

	for (unsigned i=1; i<PYRAMID_SIZE; i++)
	{
		unsigned char* tmp = (unsigned char*)malloc(getPyrWidth(i)*getPyrHeight(i)*sizeof(unsigned char));
		_current::getIoFrame8(tmp, i, getPyrWidth(i), getPyrHeight(i));

		for (unsigned y=0; y<getPyrHeight(i); y++)
			for (unsigned x=0; x<getPyrWidth(i); x++)
			{
				target[(y+shy)*width + x+shx] = tmp[y*getPyrWidth(i)+x];
			}

		free(tmp);

		if (i % 2 == 1)
			shx += getPyrWidth(i);
		else
			shy += getPyrHeight(i);
	}
}

void kanade8to24(unsigned char* target, unsigned char* src, unsigned width, unsigned height)
{
	for (unsigned y=0; y<height; y++)
		for (unsigned x=0; x<width; x++)
		{
			unsigned pos = y*width+x;
			target[3*pos] = src[pos];
			target[3*pos+1] = src[pos];
			target[3*pos+2] = src[pos];
		}
}