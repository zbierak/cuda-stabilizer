#include "kanade.h"
#include <string.h>
#include <stdlib.h>
#include <string>
#include <sstream>

#include <math.h>

namespace kcpu {

unsigned char* ioFrame24 = NULL;									// pamiec pod ramke 24b - uzywana na wejsciu i wyjsciu
unsigned char* ioFrame8[PYRAMID_SIZE] = { NULL };					// pamiec pod piramide ramke 8b  - uzywana na wejsciu jako tymczasowa dla tekstury
unsigned char* prevFrame8[PYRAMID_SIZE] = { NULL };					// poprzednia piramida ramek w formacie osmiobitowym - do wyliczania dt

float* devDx = NULL;												// pamiec na skladowe dx, dy i dt (w tej implementacji w zasadzie niepotrzebna
float* devDy = NULL;												// ale przez to prosciej sprawdzac poprawnosc implementacji GPU). No i chyba szybsza impl.
float* devDt = NULL;												

template <typename T>
inline std::string toString(T val)
{
	std::stringstream sout;
	std::string res;
	sout << val;
	sout >> res;
	return res;
}


void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height)
{
	ioFrame24 = pixels;

	for (unsigned i=0; i<PYRAMID_SIZE; ++i)
	{
		if (ioFrame8[i] != NULL)
		{
			/*
			if (prevFrame8[i] != NULL)
				free(prevFrame8[i]);

			prevFrame8[i] = ioFrame8[i];			
			*/

			if (prevFrame8[i] == NULL)
			{
				prevFrame8[i] = ioFrame8[i];
				ioFrame8[i] = NULL;
			}
		}

		ioFrame8[i] = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	}

	kanade24to8(ioFrame8[0], ioFrame24, width, height);
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

	if (devDt != NULL)
		free(devDt);
	if (devDx != NULL)
		free(devDx);
	if (devDy != NULL)
		free(devDy);
}

void buildPyramidLevel(unsigned char** pyramid, unsigned levelId, unsigned newWidth, unsigned newHeight)
{
	for (unsigned y=0; y<newHeight; y++)
		for (unsigned x=0; x<newWidth; x++)
		{
			int width = 2*newWidth;
			int np = 2*(y*width + x);
			
			unsigned color = ((unsigned)pyramid[levelId-1][np] + 
							  (unsigned)pyramid[levelId-1][np+1] + 
							  (unsigned)pyramid[levelId-1][np+width] + 
							  (unsigned)pyramid[levelId-1][np+width+1]) / 4;

			if (color > 255) 
				color = 255;

			pyramid[levelId][y*newWidth+x] = (unsigned char) color;
		}
}

void kanadeBuildPyramidLevel(unsigned levelId, unsigned newWidth, unsigned newHeight) 
{
	buildPyramidLevel(ioFrame8, levelId, newWidth, newHeight);
}

// for simplicity of function kanadeTranslate
struct color 
{
	unsigned char r;
	unsigned char g;
	unsigned char b;

	unsigned char& operator[] (unsigned int i)
	{ 
		if (i == 0) return r;
		else if (i==1) return g;
		return b;
	}
};

void kanadeTranslate(unsigned char* target, float vx, float vy, unsigned width, unsigned height)
{
	// this implementation is the reverse of the gpu, one had to be taken
	vx = -vx;
	vy = -vy; 

	int dx = (int) floor(vx);
	int dy = (int) floor(vy);

	double nx = vx - floor(vx);					// nx - fraction of the next pixel in x taken into interpolation
	double ny = vy - floor(vy);
	double tx = 1 - nx;							// tx - fraction of thix pixel taken into interpolation
	double ty = 1 - ny;

	color* frame = (color*)ioFrame24;

	for (unsigned x=0; x<width; x++)
		for (unsigned y=0; y<height; y++)
		{
			unsigned dpos = y * width + x;
			if (x+dx < 0 || y+dy < 0 || x+dx >= width-1 || y+dy >= height-1)
			{
				// take the original pixel
				//for (unsigned c=0; c<3; c++)
				//	target[3*dpos+c] = ioFrame24[3*dpos+c];
				// take black pixel
				for (unsigned c=0; c<3; c++)
					target[3*dpos+c] = 0;
			}
			else
			{
				unsigned spos = (y + dy) * width + x + dx;
				for (unsigned c=0; c<3; c++)
					target[3*dpos+c] = (unsigned char)((tx * frame[spos][c] + nx * frame[spos + 1][c]) / 2.0f
									        + (ty * frame[spos][c] + ny * frame[spos + width][c]) / 2.0f);
			}
		}

	// for building prevPyramid
	memcpy(ioFrame24, target, 3*width*height*sizeof(unsigned char));
}

void translateGrayscale(unsigned char* target, float vx, float vy, unsigned pyrLvl, unsigned width, unsigned height)
{
	int dx = (int) floor(vx);
	int dy = (int) floor(vy);

	double nx = vx - floor(vx);				// nx - fraction of the next pixel in x taken into interpolation
	double ny = vy - floor(vy);
	double tx = 1 - nx;						// tx - fraction of thix pixel taken into interpolation
	double ty = 1 - ny;

	for (unsigned x=0; x<width; x++)
		for (unsigned y=0; y<height; y++)
		{
			unsigned dpos = y * width + x;
			if (x+dx < 0 || y+dy < 0 || x+dx >= width-1 || y+dy >= height-1)
			{
				// zero if not inside the picture
				target[dpos] = 0; 
			}
			else
			{
				unsigned spos = (y + dy) * width + x + dx;
				target[dpos] = (unsigned char)((tx * ioFrame8[pyrLvl][spos] + nx * ioFrame8[pyrLvl][spos + 1]) / 2.0
									   + (ty * ioFrame8[pyrLvl][spos] + ny * ioFrame8[pyrLvl][spos + width]) / 2.0);
			}
		}
}

// subtracts images c = a-b 
void subtract(float* c, unsigned char* a, unsigned char* b, unsigned width, unsigned height)
{
	for (unsigned y=0; y<height; y++)
	{
		unsigned yr = y*width; 
		for (unsigned x=0; x<width; x++)
		{
			unsigned p = yr+x;
			c[p] = (float)a[p] - (float)b[p];
		}
	}
}

template <typename T1, typename T2>
float dotSum(T1* a, T2* b, unsigned size)
{
	float sum = 0.0;
	for (unsigned i=0; i<size; i++)
		sum += a[i] * b[i];
	return sum;
}

// for window operations
template <typename T1, typename T2>
float dotSum(T1* a, T2* b, int width, int height)
{
	float sum = 0.0f;
	int px = width/2;
	int py = height/2;

	// in case window is greater than our image (well, it happens)
	int sx = std::max(0, px-WINDOW_SIZE_DIV_2);
	int ex = std::min(width-1, px+WINDOW_SIZE_DIV_2);
	int sy = std::max(0, py-WINDOW_SIZE_DIV_2);
	int ey = std::min(height-1, py+WINDOW_SIZE_DIV_2);

	for (int y=sy; y<=ey; y++)
		for (int x=sx; x<=ex; x++)
		{
			int pos = y*width+x;
			sum += a[pos] * b[pos];
		}

	return sum;
}

void kanadeCalculateG(unsigned pyrLvl, unsigned width, unsigned height, float& dxdx, float& dxdy, float& dydy) 
{
	if (devDx != NULL)
		free(devDx);
	if (devDy != NULL)
		free(devDy);

	devDx = (float*) malloc(width*height*sizeof(float));
	devDy = (float*) malloc(width*height*sizeof(float));
	memset(devDx, 0, width*height*sizeof(float));
	memset(devDy, 0, width*height*sizeof(float));

	for (unsigned y=0; y<height; y++)
		for (unsigned x=0; x<width; x++)
		{
			unsigned idx = y * width + x;
			if (x >= 1 && x < width-1)
				devDx[idx] = ((float)prevFrame8[pyrLvl][idx+1] - (float)prevFrame8[pyrLvl][idx-1]) / 2.0f;		
			if (y >= 1 && y < height-1)
				devDy[idx] = ((float)prevFrame8[pyrLvl][idx+width] - (float)prevFrame8[pyrLvl][idx-width]) / 2.0f;
		}

	unsigned size = width*height;
	dxdx = dotSum(devDx, devDx, size);
	dxdy = dotSum(devDx, devDy, size);
	dydy = dotSum(devDy, devDy, size);
	//dxdx = dotSum(devDx, devDx, width, height);
	//dxdy = dotSum(devDx, devDy, width, height);
	//dydy = dotSum(devDy, devDy, width, height);
}

void kanadeCalculateB(unsigned pyrLvl, float vx, float vy, unsigned width, unsigned height, float& bx, float& by) 
{
	if (devDt != NULL)
		free(devDt);

	devDt = (float*) malloc(width*height*sizeof(float));
	unsigned char* tmp = (unsigned char*) malloc(width*height*sizeof(unsigned char));

	translateGrayscale(tmp, vx, vy, pyrLvl, width, height);
	subtract(devDt, prevFrame8[pyrLvl], tmp, width, height);
	free(tmp);

	unsigned size = width*height;
	bx = dotSum(devDx, devDt, size);
	by = dotSum(devDy, devDt, size);
	//bx = dotSum(devDx, devDt, width, height);
	//by = dotSum(devDy, devDt, width, height);
}

void kanadePrepareForNextFrame(unsigned width[PYRAMID_SIZE], unsigned height[PYRAMID_SIZE])
{
	kanade24to8(prevFrame8[0], ioFrame24, width[0], height[0]);

	for (unsigned lvl=1; lvl<PYRAMID_SIZE; lvl++)
		buildPyramidLevel(prevFrame8, lvl, width[lvl], height[lvl]);
}

// for testing purposes
void getIoFrame8(unsigned char* target, unsigned lvl, unsigned width, unsigned height) 
{
	memcpy(target, ioFrame8[lvl], width*height*sizeof(unsigned char));
}

void getPrevFrame8(unsigned char* target, unsigned lvl, unsigned width, unsigned height) 
{
	memcpy(target, prevFrame8[lvl], width*height*sizeof(unsigned char));
}

void getDevDx(float* target, unsigned width, unsigned height)
{
	memcpy(target, devDx, width*height*sizeof(float));
}

void getDevDy(float* target, unsigned width, unsigned height)
{
	memcpy(target, devDy, width*height*sizeof(float));
}

void getDevDt(float* target, unsigned width, unsigned height)
{
	memcpy(target, devDt, width*height*sizeof(float));
}

}