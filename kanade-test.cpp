#include "kanade.h"
#include "kanade-inner.h"
#include "kanade-test.h"

#include <stdlib.h>
#include <math.h>

#include <iostream>

/*
 * How to execute tests:
 * 1) singular tests compare whether the given implementation produces (any) results 
 *    and leaves off the analysis to the user. Example of such is kanadeTestBuildPyramid()
 *    While executing such functions, the implementation will be taken from the namespace
 *    defined by macro "_current"
 * 2) comparative tests compare results produced by two different methods and display
 *    obtained errors. To select the implementations, properly define "_cmp1" and "_cmp2" 
 8    macros
 */

#ifdef KANADE_NO_GPU
#define _current kcpu
#define _cmp1 kcpu
#define _cmp2 kcpu
#else
#define _current kgpu
#define _cmp1 kgpu
#define _cmp2 kcpu
#endif

void kanadeTestNextFrame(unsigned char* pixels, unsigned width, unsigned height)
{
	_cmp1::kanadeNextFrame(pixels, width, height);
	_cmp2::kanadeNextFrame(pixels, width, height);
}

// buduje piramide z POPRZEDNICH ramek (bo daje to nam wiecej informacji)
void kanadeTestBuildPyramid(unsigned char* target, unsigned width, unsigned height)
{
	buildPyramid(width, height);
	_current::getPrevFrame8(target, 0, width, height);
	
	unsigned shx = 0;
	unsigned shy = 0;

	for (unsigned i=1; i<PYRAMID_SIZE; i++)
	{
		unsigned char* tmp = (unsigned char*)malloc(getPyrWidth(i)*getPyrHeight(i)*sizeof(unsigned char));
		_current::getPrevFrame8(tmp, i, getPyrWidth(i), getPyrHeight(i));

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

// if everything is correct, this test should return a black frame (every pixel should be exactly 0)
void kanadeTestCompareBuildPyramid(unsigned char* target, unsigned width, unsigned height)
{
	buildPyramid(width, height);
	for (unsigned i=1; i<PYRAMID_SIZE; i++) 
	{
		_cmp1::kanadeBuildPyramidLevel(i, getPyrWidth(i), getPyrHeight(i));
		_cmp2::kanadeBuildPyramidLevel(i, getPyrWidth(i), getPyrHeight(i));
	}
	
	unsigned char* t1 = (unsigned char*)malloc(width*height*sizeof(unsigned char));
	unsigned char* t2 = (unsigned char*)malloc(width*height*sizeof(unsigned char));

	_cmp1::getIoFrame8(t1, 0, width, height);
	_cmp2::getIoFrame8(t2, 0, width, height);

	for (unsigned i=0; i<width*height; i++)
		target[i] = abs(t1[i]-t2[i]);

	free(t1);
	free(t2);
	
	unsigned shx = 0;
	unsigned shy = 0;

	for (unsigned i=1; i<PYRAMID_SIZE; i++)
	{
		t1 = (unsigned char*)malloc(getPyrWidth(i)*getPyrHeight(i)*sizeof(unsigned char));
		t2 = (unsigned char*)malloc(getPyrWidth(i)*getPyrHeight(i)*sizeof(unsigned char));
		_cmp1::getIoFrame8(t1, i, getPyrWidth(i), getPyrHeight(i));
		_cmp2::getIoFrame8(t2, i, getPyrWidth(i), getPyrHeight(i));

		for (unsigned y=0; y<getPyrHeight(i); y++)
			for (unsigned x=0; x<getPyrWidth(i); x++)
			{
				target[(y+shy)*width + x+shx] = abs(t1[y*getPyrWidth(i)+x] - t2[y*getPyrWidth(i)+x]);
			}

		free(t1);
		free(t2);

		if (i % 2 == 1)
			shx += getPyrWidth(i);
		else
			shy += getPyrHeight(i);
	}
}

void kanadeTestGenerateG(unsigned char* target, unsigned width, unsigned height)
{
	buildPyramid(width, height);
	for (unsigned i=1; i<PYRAMID_SIZE; i++) 
	{
		_cmp1::kanadeBuildPyramidLevel(i, getPyrWidth(i), getPyrHeight(i));
		_cmp2::kanadeBuildPyramidLevel(i, getPyrWidth(i), getPyrHeight(i));
	}

	for (unsigned i=0; i<PYRAMID_SIZE; i++)
	{
		float* t1 = (float*)malloc(getPyrWidth(i)*getPyrHeight(i)*sizeof(float));
		float* t2 = (float*)malloc(getPyrWidth(i)*getPyrHeight(i)*sizeof(float));

		unsigned size = getPyrWidth(i) * getPyrHeight(i);

		float dxdx1, dxdy1, dydy1;
		float dxdx2, dxdy2, dydy2;
		float errx = 0.0f, erry = 0.0f;

		_cmp1::kanadeCalculateG(i, getPyrWidth(i), getPyrHeight(i), dxdx1, dxdy1, dydy1);
		_cmp2::kanadeCalculateG(i, getPyrWidth(i), getPyrHeight(i), dxdx2, dxdy2, dydy2);

		_cmp1::getDevDx(t1, getPyrWidth(i), getPyrHeight(i));
		_cmp2::getDevDx(t2, getPyrWidth(i), getPyrHeight(i));
		
		for (unsigned pos=0; pos<size; pos++)
		{
			float diff = abs(t1[pos] - t2[pos]);
			if (diff > errx)
				errx = diff;

		}

		_cmp1::getDevDy(t1, getPyrWidth(i), getPyrHeight(i));
		_cmp2::getDevDy(t2, getPyrWidth(i), getPyrHeight(i));

		for (unsigned pos=0; pos<size; pos++)
		{
			float diff = abs(t1[pos] - t2[pos]);
			if (diff > erry)
				erry = diff;
		}

		free(t1);
		free(t2);

		std::cout << "Pyramid level " << i << ",\tmax dx error=" << errx << ",\tmax dy error=" << erry << std::endl;
		std::cout << "\t|dxdx| = " << abs(dxdx1 - dxdx2) << "\t|dxdy| = " << abs(dxdy1 - dxdy2) << "\t|dydy| = " << abs(dydy1 - dydy2) << std::endl;
	}
}

// maksymalny blad wyszedl mi 1, ale nie zastanawialem sie zbytnio dlaczemu nie jest rowny zeru
// w pierwszej ramce pojawia sie wiekszy blad w skladowych piramidy, ale to dlatego, ze nie sa pewnie jeszcze wyliczone
// dobry wynik tej funkcji to czarny obraz w zmiennej target
void kanadeTestGenerateB(unsigned char* target, unsigned width, unsigned height)
{
	const float vx = 10.5f;
	const float vy = 20.3f;

	buildPyramid(width, height);
	for (unsigned i=1; i<PYRAMID_SIZE; i++) 
	{
		_cmp1::kanadeBuildPyramidLevel(i, getPyrWidth(i), getPyrHeight(i));
		_cmp2::kanadeBuildPyramidLevel(i, getPyrWidth(i), getPyrHeight(i));
	}

	for (unsigned i=0; i<PYRAMID_SIZE; i++)
	{
		float* t1 = (float*)malloc(getPyrWidth(i)*getPyrHeight(i)*sizeof(float));
		float* t2 = (float*)malloc(getPyrWidth(i)*getPyrHeight(i)*sizeof(float));

		unsigned size = getPyrWidth(i) * getPyrHeight(i);

		float dxdx1, dxdy1, dydy1, bx1, by1;
		float dxdx2, dxdy2, dydy2, bx2, by2;
		float err = 0.0f;

		// to jest potrzebne, poniewaz B korzysta z dx i dy
		_cmp1::kanadeCalculateG(i, getPyrWidth(i), getPyrHeight(i), dxdx1, dxdy1, dydy1);
		_cmp2::kanadeCalculateG(i, getPyrWidth(i), getPyrHeight(i), dxdx2, dxdy2, dydy2);

		_cmp1::kanadeCalculateB(i, vx, vy, getPyrWidth(i), getPyrHeight(i), bx1, by1);
		_cmp2::kanadeCalculateB(i, vx, vy, getPyrWidth(i), getPyrHeight(i), bx2, by2);

		_cmp1::getDevDt(t1, getPyrWidth(i), getPyrHeight(i));
		_cmp2::getDevDt(t2, getPyrWidth(i), getPyrHeight(i));
		
		for (unsigned pos=0; pos<size; pos++)
		{
			float diff = abs(t1[pos] - t2[pos]);
			if (diff > err)
				err = diff;
		}

		if (i==0)
		{
			for (unsigned y=0; y<getPyrHeight(i); y++)
				for (unsigned x=0; x<getPyrWidth(i); x++)
				{
					unsigned pos = y*getPyrWidth(i)+x;
					target[pos] = abs(t1[pos] - t2[pos]);
				}
		}

		free(t1);
		free(t2);

		std::cout << "Pyramid level " << i << ",\tmax dt error=" << err << std::endl;
		std::cout << "\t|bx| = " << abs(bx1 - bx2) << "\t|by| = " << abs(by1 - by2) << std::endl;
	}
}