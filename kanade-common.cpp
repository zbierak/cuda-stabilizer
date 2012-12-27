#include "kanade.h"
#include "kanade-inner.h"

#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "timer.h"

Timer totalTimer;
unsigned frameCount;

double gTime, bTime, estTime, pyrTime, transTime; 

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
	Timer t;

	_current::kanadeNextFrame(pixels, width, height);	
	buildPyramid(width, height);

	pyrTime += t.stop();
}

void executeOnPyramidLevel(unsigned lvl, float& vx, float& vy)
{
	float nx = 0.0f, ny = 0.0f;	
	float dxdx, dxdy, dydy;
	float bx, by;

	//   G matrix:
	// [ dxdx dxdy ]
	// [ dxdy dydy ]
	Timer gt;
	_current::kanadeCalculateG(lvl, getPyrWidth(lvl), getPyrHeight(lvl), dxdx, dxdy, dydy);
	gTime += gt.stop();

	// czy macierz G jest osobliwa
	float delta = (dxdx+dydy)*(dxdx+dydy) - 4*(dxdx*dydy-dxdy*dxdy);
	float l1 = (-(dxdx+dydy) - sqrt(delta))/2;
	float l2 = (-(dxdx+dydy) + sqrt(delta))/2;

	//std::cout << "L=" << lvl << "\tdxdx=" << dxdx << "\tdxdy=" << dxdy << "\tdydy=" << dydy << std::endl;

	for (unsigned it=0; it<ITERATION_COUNT; ++it)
	{
		Timer bt;
		_current::kanadeCalculateB(lvl, vx, vy, getPyrWidth(lvl), getPyrHeight(lvl), bx, by);
		bTime += bt.stop();

		// Optical flow (Lucas-Kanade) nk = G^-1 * bk
		if (l1<-1 && l2<-1)
		{
			float det = dxdx*dydy - dxdy*dxdy;
			nx = (dydy * bx - dxdy * by) / det;
			ny = (-dxdy * bx + dxdx * by) / det;
		}
		else if (l1<-1 && l2>=-1)
		{
			nx = -bx/(dydy+dxdx);
			ny = -by/(dydy+dxdx);
		}
		else
		{
			nx = 0; 
			ny = 0;
		}

		vx += nx;
		vy += ny;
	}
}

void kanadeExecute(unsigned char* target, unsigned width, unsigned height)
{
	float vx = 0.0f, vy = 0.0f;	

	Timer estT;
	for (int lvl = PYRAMID_SIZE-1; lvl >= 0; --lvl)
	{
		executeOnPyramidLevel(lvl, vx, vy);

		// starting point for the next level
		if (lvl > 0)
		{
			vx = 2*vx;
			vy = 2*vy;
		}
	}
	estTime += estT.stop();

	//std::cout << "Vx: " << vx << " Vy: " << vy << std::endl;

	Timer transT;
	_current::kanadeTranslate(target, -vx, -vy, width, height);
	transTime += transT.stop();

	#ifndef KANADE_NO_RFRAME_UPDATE
	Timer pfnf;
	_current::kanadePrepareForNextFrame(pyrWidth, pyrHeight);
	pyrTime += pfnf.stop();
	#endif

	frameCount++;
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

	// level zero is already in memory
	for (unsigned i=1; i<PYRAMID_SIZE; i++)
		_current::kanadeBuildPyramidLevel(i, pyrWidth[i], pyrHeight[i]);
}

void kanadeInit()
{
	kcpu::kanadeInit();

	#ifndef KANADE_NO_GPU
	kgpu::kanadeInit();
	#endif

	gTime = 0;
	bTime = 0;
	estTime = 0;
	pyrTime = 0;
	transTime = 0;
	frameCount = 0;

	totalTimer.restart();
}

void kanadeCleanup()
{
	kcpu::kanadeCleanup();

	#ifndef KANADE_NO_GPU
	kgpu::kanadeCleanup();
	#endif
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

void kanade24to8(unsigned char* target, unsigned char* src, unsigned width, unsigned height)
{
	for (unsigned y=0; y<height; y++)
		for (unsigned x=0; x<width; x++)
		{
			unsigned pos = y*width+x;
			//float color = floor(0.299f * (float)src[3*pos] + 0.587f * (float)src[3*pos+1] + 0.114f * (float)src[3*pos+2]);	// ten sposob powodowal bledy miedzy implementacja CPU a GPU
			//float color = floor((src[3*pos] + src[3*pos+1] + src[3*pos+2])/3.0f);
			//if (color > 255.0f) color = 255.0f;
			target[pos] = max(max(src[3*pos], src[3*pos+1]), src[3*pos+2]);
		}
}

void kanadePrintStats()
{
	double tt = totalTimer.stop();
	double other = tt - pyrTime - estTime - transTime;

	std::cout << std::endl;
	std::cout << "Total execution time: " << tt << "s (" << tt/frameCount << " s per frame, " << frameCount / tt << " fps), out of which:" << std::endl;
	std::cout << "    frame loading & pyr building: " << pyrTime << "s (" << (int)(100*pyrTime/tt) << "%)" << std::endl;
	std::cout << "    estimation:                   " << estTime << "s (" << (int)(100*estTime/tt) << "%), out of which:" << std::endl;
	std::cout << "        g matrix computation:     " << gTime << "s (" << (int)(100*gTime/estTime) << "%)" << std::endl;
	std::cout << "        b vector computation:     " << bTime << "s (" << (int)(100*bTime/estTime) << "%)" << std::endl;
	std::cout << "    translation:                  " << transTime << "s (" << (int)(100*transTime/tt) << "%)" << std::endl;
	std::cout << "    other (frame encoding etc):   " << other << "s (" << (int)(100*other/tt) << "%)" << std::endl;
}

// undefine selectors
#undef _current