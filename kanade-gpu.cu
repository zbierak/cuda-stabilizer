#ifndef KANADE_NO_GPU

namespace kgpu {

const unsigned BLOCK_2POW = 5;										
const unsigned BLOCK_DIM = 2 << (BLOCK_2POW-1);
//const unsigned BLOCK_DIM_SQ = BLOCK_DIM * BLOCK_DIM;

const unsigned RED_BLOCK_2POW = 7;									// BLOCK_SIZE = 2 ^ BLOCK_2POW
const unsigned RED_BLOCK_DIM = 2 << (RED_BLOCK_2POW-1);

#include "kanade.h"
#include "helper_cuda.h"

unsigned char* ioFrame24 = NULL;									// pamiec pod ramke 24b - uzywana na wejsciu i wyjsciu
unsigned char* ioFrame8[PYRAMID_SIZE] = { NULL };					// pamiec pod piramide ramke 8b  - uzywana na wejsciu jako tymczasowa dla tekstury
unsigned char* prevFrame8[PYRAMID_SIZE] = { NULL };					// poprzednia piramida ramek w formacie osmiobitowym - do wyliczania dt

unsigned char* gpuFrame;											// ramka oryginalna, ktora bedzie uzywana do przesuwania

float* devDx = NULL;												// pamiec na skladowe dx, dy i dt
float* devDy = NULL;
float* devDt = NULL;

float* cpuDx = NULL;
float* cpuDy = NULL;
float* cpuDt = NULL;

float* devG = NULL;
float* devB = NULL;
float* cpuG = NULL;
float* cpuB = NULL;

template <typename T>
__device__ inline unsigned char toColor(T value)
{
	if (value > 255) return 255;
	if (value < 0) return 0;
	return (unsigned char) value;
}

template <>
__device__ inline unsigned char toColor<unsigned>(unsigned value)
{
	if (value > 255) return 255;
	return (unsigned char) value;
}

/*
// shared memory version with coalescing - works slower than the second one
__global__ void toGrayscale(unsigned char* frame24, unsigned char* frame8, unsigned width, unsigned height)
{
	__shared__ float colors[3*BLOCK_DIM];

	int pos = blockIdx.x*blockDim.x + threadIdx.x;
	colors[threadIdx.x] = frame24[pos];
	colors[threadIdx.x + BLOCK_DIM] = frame24[pos+BLOCK_DIM];
	colors[threadIdx.x + 2*BLOCK_DIM] = frame24[pos+2*BLOCK_DIM];

	__syncthreads();

	if (pos >= width*height) return;

	unsigned char r = colors[3*threadIdx.x];
	unsigned char g = colors[3*threadIdx.x+1];
	unsigned char b = colors[3*threadIdx.x+2];

	frame8[pos] = MAX(MAX(r, g), b);		
}
*/

__global__ void toGrayscale(unsigned char* frame24, unsigned char* frame8, unsigned width, unsigned height)
{
	// this version has limited coalescing, but works faster than the one with shared memory

	int pos = blockIdx.x*blockDim.x + threadIdx.x;
	if (pos >= width*height) return;

	unsigned char r = frame24[3*pos];
	unsigned char g = frame24[3*pos+1];
	unsigned char b = frame24[3*pos+2];

	// konwersja na skale szarosci wg. jasnosci piksela
	//frame8[pos] = toColor(floor(0.299f * r + 0.587 * g + 0.114 * b));		// ten sposob (przeksztalcenie na YUV) powodowal bledy miedzy implementacja CPU a GPU
	//frame8[pos] = toColor(floor((r + g + b)/3.0f));
	frame8[pos] = MAX(MAX(r, g), b);	
}

__global__ void build_pyramid_level(unsigned char* prevLvl8b, unsigned char* newLvl8b, unsigned newWidth, unsigned newHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// zezwol tylko na wartosci z poprawnego zakresu
	if (x >= newWidth || y >= newHeight) return;

	// srednia z czterech wartosci, mozna tez zrobic min/max
	// @todo: to mozna zrobic wydajniej laczac operacje odczytu

	int width = 2*newWidth;
	int np = 2*(y*width + x);
	newLvl8b[y*newWidth+x] = toColor(
		((unsigned)prevLvl8b[np] + (unsigned)prevLvl8b[np+1] + 
		 (unsigned)prevLvl8b[np+width] + (unsigned)prevLvl8b[np+width+1]) / 4);
}

__global__ void translate(float vx, float vy, unsigned char* input24, unsigned char* output24, unsigned width, unsigned height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	// zezwol tylko na wartosci z poprawnego zakresu
	if (x >= width || y >= height) return;

	// nearest neighbour version
	int dx = (int) round(vx);					
	int dy = (int) round(vy);

	unsigned dpos = 3*(y * width + x);

	if (x-dx >= 0 && y-dy >= 0 && x-dx < width-1 && y-dy < height-1)
	{
		unsigned spos = 3*((y - dy) * width + x - dx);
		for (unsigned c=0; c<3; c++) 
		{
			output24[dpos+c] = input24[spos+c];
		}
	}
}

__global__ void calculate_dxdy(unsigned char* frame8, float* dx, float* dy, unsigned width, unsigned height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	// zezwol tylko na wartosci z poprawnego zakresu
	if (x >= width || y >= height) return;

	// rozwazyc czy nie zastapic tego dodaniem paddingu po obu stronach obrazu

	// @todo: to mozna zrobic wydajniej przy uzyciu pamieci dzielonej
	int pos = y*width + x;
	if (x != 0 && x != width-1)
		dx[pos] = ((float)(frame8[pos+1]-frame8[pos-1])) / 2.0f;

	if (y != 0 && y != height-1)
		dy[pos] = ((float)(frame8[pos+width]-frame8[pos-width])) / 2.0f;
}

__global__ void calculate_dt(unsigned pyrLvl, float vx, float vy, unsigned char* prevFrame8b, unsigned char* currFrame8b, float* dt, unsigned width, unsigned height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	// zezwol tylko na wartosci z poprawnego zakresu
	if (x >= width || y >= height) return;

	int dx = (int) floor(vx);
	int dy = (int) floor(vy);

	float nx = vx - floor(vx);					// fractions of the next pixel taken into interpolation
	float ny = vy - floor(vy);
	float tx = 1 - nx;							// fraction of this pixel taken into interpolation
	float ty = 1 - ny;
	
	float interpolated = 0;
	if (x+dx >= 0 && y+dy >= 0 && x+dx < width-1 && y+dy < height-1)
	{
		unsigned spos = (y + dy) * width + x + dx;

		float y1 = (tx * currFrame8b[spos] + nx * currFrame8b[spos+1]);
		float y2 = (tx * currFrame8b[spos+width] + nx * currFrame8b[spos+width+1]);

		interpolated = (unsigned char)(ty * y1 + ny * y2);
	}

	int dpos = y*width + x;
	dt[dpos] = (float)prevFrame8b[dpos] - interpolated;
}

/*
// an approach to shared memory - slower than the previous, only partly coalesced implementation
__global__ void calculate_dt(unsigned pyrLvl, float vx, float vy, unsigned char* prevFrame8b, unsigned char* currFrame8b, float* dt, unsigned width, unsigned height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	__shared__ float lo_colors[2*BLOCK_DIM_SQ];
	__shared__ float hi_colors[2*BLOCK_DIM_SQ];
	
	// zezwol tylko na wartosci z poprawnego zakresu
	if (x >= width || y >= height) return;

	int dx = (int) floor(vx);
	int dy = (int) floor(vy);

	float nx = vx - floor(vx);					// fractions of the next pixel taken into interpolation
	float ny = vy - floor(vy);
	float tx = 1 - nx;							// fraction of this pixel taken into interpolation
	float ty = 1 - ny;
	
	float interpolated = 0;
	if (x+dx >= 0 && y+dy >= 0 && x+dx < width-1 && y+dy < height-1)
	{
		unsigned spos = (y + dy) * width + x + dx;
		unsigned bpos = threadIdx.y*blockDim.x + threadIdx.x;

		lo_colors[bpos] = currFrame8b[spos];
		lo_colors[bpos+BLOCK_DIM_SQ] = currFrame8b[spos+BLOCK_DIM_SQ];
		hi_colors[bpos] = currFrame8b[spos+width];
		hi_colors[bpos+BLOCK_DIM_SQ] = currFrame8b[spos+width+BLOCK_DIM_SQ];

		__syncthreads();

		float y1 = tx * lo_colors[bpos] + nx * lo_colors[bpos+1];
		float y2 = tx * hi_colors[bpos] + nx * hi_colors[bpos+1];

		interpolated = (unsigned char)(ty * y1 + ny * y2);
	}

	int dpos = y*width + x;
	dt[dpos] = (float)prevFrame8b[dpos] - interpolated;
}
*/

__global__ void reduce_to_g(float* dx, float* dy, float* dxdx, float* dxdy, float* dydy, unsigned size)
{
	__shared__ float t_dxdx[RED_BLOCK_DIM];
	__shared__ float t_dxdy[RED_BLOCK_DIM];
	__shared__ float t_dydy[RED_BLOCK_DIM];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	float l_dx = dx[idx];
	float l_dy = dy[idx];

	t_dxdx[threadIdx.x] = l_dx * l_dx;
	t_dxdy[threadIdx.x] = l_dx * l_dy;
	t_dydy[threadIdx.x] = l_dy * l_dy;

	__syncthreads();

	int half = blockDim.x;

	#pragma unroll
	for (int i=0; i<RED_BLOCK_2POW; i++)
	{
		half = half >> 1;

		if (threadIdx.x < half)
		{
			int thread2 = threadIdx.x + half;
			t_dxdx[threadIdx.x] +=  t_dxdx[thread2];
			t_dxdy[threadIdx.x] +=  t_dxdy[thread2];
			t_dydy[threadIdx.x] +=  t_dydy[thread2];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) 
	{
		atomicAdd(dxdx, t_dxdx[0]);
		atomicAdd(dxdy, t_dxdy[0]);
		atomicAdd(dydy, t_dydy[0]);
	}
}

__global__ void reduce_to_b(float* dx, float* dy, float* dt, float* bx, float* by)
{
	__shared__ float t_dxdt[RED_BLOCK_DIM];
	__shared__ float t_dydt[RED_BLOCK_DIM];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	float l_dt = dt[idx];

	t_dxdt[threadIdx.x] = dx[idx] * l_dt;
	t_dydt[threadIdx.x] = dy[idx] * l_dt;

	__syncthreads();

	int half = blockDim.x;

	#pragma unroll
	for (int i=0; i<RED_BLOCK_2POW; i++)
	{
		half = half >> 1;

		if (threadIdx.x < half)
		{
			int thread2 = threadIdx.x + half;
			t_dxdt[threadIdx.x] +=  t_dxdt[thread2];
			t_dydt[threadIdx.x] +=  t_dydt[thread2];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) 
	{
		atomicAdd(bx, t_dxdt[0]);
		atomicAdd(by, t_dydt[0]);
	}
}

/*
 * w wiekszosci przypadkow pamiec alokowana jest tylko raz na caly film, ale np. ioFrame8 jest alokowane co klatke
 * (po to, zeby nie kopiowac zawartosci poprzedniej ramki, przypisujemy jej pointer, a pod nowa ramke alokujemy
 * nowa pamiec)
 */

void allocateMemoryIfNeeded(unsigned width, unsigned height)
{
	if (ioFrame24 == NULL)
		checkCudaErrors(cudaMalloc(&ioFrame24, 3 * width * height * sizeof(unsigned char))); 

	for (unsigned i=0; i<PYRAMID_SIZE; ++i)
	{
		if (ioFrame8[i] == NULL)
			checkCudaErrors(cudaMalloc(&ioFrame8[i], width * height * sizeof(unsigned char))); 
	}
		
	if (gpuFrame == NULL)
		checkCudaErrors(cudaMalloc(&gpuFrame, 3 * width * height * sizeof(unsigned char))); 

	unsigned wthAligned = width * height;
	if (wthAligned % RED_BLOCK_DIM != 0)
		wthAligned += (RED_BLOCK_DIM - wthAligned % RED_BLOCK_DIM);

	if (devDx == NULL)
		checkCudaErrors(cudaMalloc(&devDx, wthAligned * sizeof(float))); 

	if (devDy == NULL)
		checkCudaErrors(cudaMalloc(&devDy, wthAligned * sizeof(float))); 

	if (devDt == NULL)
		checkCudaErrors(cudaMalloc(&devDt, wthAligned * sizeof(float))); 

	if (cpuDx == NULL)
		cpuDx = (float*) malloc(width * height * sizeof(float));

	if (cpuDy == NULL)
		cpuDy = (float*) malloc(width * height * sizeof(float));

	if (cpuDt == NULL)
		cpuDt = (float*) malloc(width * height * sizeof(float));	
}

void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height)
{
	for (unsigned i=0; i<PYRAMID_SIZE; ++i)
	{
		if (ioFrame8[i] != NULL)
		{			
			if (prevFrame8[i] == NULL)
			{
				prevFrame8[i] = ioFrame8[i];
				ioFrame8[i] = NULL;
			}
		}
	}

	allocateMemoryIfNeeded(width, height);

	// przygotuj dane wejsciowe pod tekstury
	checkCudaErrors(cudaMemcpy(ioFrame24, pixels, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
	toGrayscale<<<(width*height + BLOCK_DIM-1)/BLOCK_DIM, BLOCK_DIM>>>(ioFrame24, ioFrame8[0], width, height);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(gpuFrame, ioFrame24, 3 * width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice));	
}

void kanadePrepareForNextFrame(unsigned width[PYRAMID_SIZE], unsigned height[PYRAMID_SIZE])
{
	/*
	 * right now we have the translated frame in ioFrame24 (as this function is called at the end of kanadeExecute())
	 * Consequently, we will use the translated frame as a reference frame for the next frame. Hence, we have to 
	 * build the pyramid (again) for the translated frame, duh. 
	 */

	// prepare pyramid level zero
	toGrayscale<<<(width[0]*height[0] + BLOCK_DIM-1)/BLOCK_DIM, BLOCK_DIM>>>(ioFrame24, prevFrame8[0], width[0], height[0]);

	// prepare other pyramid levels
	for (unsigned i=1; i<PYRAMID_SIZE; i++)
	{
		dim3 dimGrid((width[i] + BLOCK_DIM - 1) / BLOCK_DIM, (height[i] + BLOCK_DIM - 1) / BLOCK_DIM);	
		dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	

		build_pyramid_level<<<dimGrid, dimBlock>>>(prevFrame8[i-1], prevFrame8[i], width[i], height[i]);
	}
}

void kanadeTranslate(unsigned char* target, float vx, float vy, unsigned width, unsigned height)
{	
	dim3 dimGrid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	

	cudaMemset(ioFrame24, 0, 3*width*height*sizeof(unsigned char));
	translate<<<dimGrid, dimBlock>>>(vx, vy, gpuFrame, ioFrame24, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(target, ioFrame24, 3 * width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
}

void kanadeCalculateG(unsigned pyrLvl, unsigned width, unsigned height, float& dxdx, float& dxdy, float& dydy)
{
	// we need to align to BLOCK_DIM, so that reduction works correctly
	unsigned sizeAligned = width * height;
	if (sizeAligned % RED_BLOCK_DIM != 0)
		sizeAligned += (RED_BLOCK_DIM - sizeAligned % RED_BLOCK_DIM);
	checkCudaErrors(cudaMemset(devDx, 0, sizeAligned * sizeof(float)));
	checkCudaErrors(cudaMemset(devDy, 0, sizeAligned * sizeof(float)));

	dim3 dimGrid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	
	checkCudaErrors(cudaMemset(devG, 0, 3 * sizeof(float)));

	calculate_dxdy<<<dimGrid, dimBlock>>>(prevFrame8[pyrLvl], devDx, devDy, width, height);
	reduce_to_g<<<sizeAligned / RED_BLOCK_DIM, RED_BLOCK_DIM>>>(devDx, devDy, &devG[0], &devG[1], &devG[2], sizeAligned);

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(cpuG, devG, 3 * sizeof(float), cudaMemcpyDeviceToHost));
	dxdx = cpuG[0];
	dxdy = cpuG[1];
	dydy = cpuG[2];
}

void kanadeCalculateB(unsigned pyrLvl, float vx, float vy, unsigned width, unsigned height, float& bx, float& by)
{
	unsigned sizeAligned = width * height;
	if (sizeAligned % RED_BLOCK_DIM != 0)
	{
		unsigned diff = (RED_BLOCK_DIM - sizeAligned % RED_BLOCK_DIM); 
		checkCudaErrors(cudaMemset(&devDt[sizeAligned], 0, diff * sizeof(float)));
		sizeAligned += diff;
		
	}

	dim3 dimGrid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	
	checkCudaErrors(cudaMemset(devB, 0, 2 * sizeof(float)));

	calculate_dt<<<dimGrid, dimBlock>>>(pyrLvl, vx, vy, prevFrame8[pyrLvl], ioFrame8[pyrLvl], devDt, width, height);	
	reduce_to_b<<<sizeAligned / RED_BLOCK_DIM, RED_BLOCK_DIM>>>(devDx, devDy, devDt, &devB[0], &devB[1]);

	checkCudaErrors(cudaMemcpy(cpuB, devB, 2 * sizeof(float), cudaMemcpyDeviceToHost));
	bx = cpuB[0];
	by = cpuB[1];
}

// level 0 is automatically initiated elsewhere
void kanadeBuildPyramidLevel(unsigned levelId, unsigned newWidth, unsigned newHeight)
{
	dim3 dimGrid((newWidth + BLOCK_DIM - 1) / BLOCK_DIM, (newHeight + BLOCK_DIM - 1) / BLOCK_DIM);	
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	

	build_pyramid_level<<<dimGrid, dimBlock>>>(ioFrame8[levelId-1], ioFrame8[levelId], newWidth, newHeight);
}

void kanadeInit()
{
	cudaDeviceProp deviceProp;
	int devID = 0;
	checkCudaErrors(cudaSetDevice(devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);	

	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

	checkCudaErrors(cudaMalloc(&devG, 3 * sizeof(float))); 
	checkCudaErrors(cudaMalloc(&devB, 2 * sizeof(float))); 
	
	cpuG = (float*) malloc(3 * sizeof(float));
	cpuB = (float*) malloc(2 * sizeof(float));	
}

void kanadeCleanup()
{
	for (unsigned i=0; i<PYRAMID_SIZE; i++)
	{
		if (ioFrame8[i] != NULL)
			cudaFree(ioFrame8[i]);
		if (prevFrame8[i] != NULL)
			cudaFree(prevFrame8[i]);
	}

	if (ioFrame24 != NULL)
		cudaFree(ioFrame24);

	if (gpuFrame != NULL)
		cudaFree(gpuFrame);

	if (devDx != NULL)
		cudaFree(devDx);

	if (devDy != NULL)
		cudaFree(devDy);

	if (devDt != NULL)
		cudaFree(devDt);

	if (cpuDx != NULL)
		free(cpuDx);
	if (cpuDy != NULL)
		free(cpuDy);
	if (cpuDt != NULL)
		free(cpuDt);

	if (devG != NULL)
		cudaFree(devG);
	if (devB != NULL)
		cudaFree(devB);
	if (cpuG != NULL)
		free(cpuG);
	if (cpuB != NULL)
		free(cpuB);

	checkCudaErrors(cudaDeviceReset());
}

// for testing purposes only
void getIoFrame8(unsigned char* target, unsigned lvl, unsigned width, unsigned height) 
{
	checkCudaErrors(cudaMemcpy(target, ioFrame8[lvl], width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
}

void getPrevFrame8(unsigned char* target, unsigned lvl, unsigned width, unsigned height) 
{
	checkCudaErrors(cudaMemcpy(target, prevFrame8[lvl], width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
}

void getDevDx(float* target, unsigned width, unsigned height)
{
	checkCudaErrors(cudaMemcpy(target, devDx, width * height * sizeof(float), cudaMemcpyDeviceToHost));
}

void getDevDy(float* target, unsigned width, unsigned height)
{
	checkCudaErrors(cudaMemcpy(target, devDy, width * height * sizeof(float), cudaMemcpyDeviceToHost));
}

void getDevDt(float* target, unsigned width, unsigned height)
{
	checkCudaErrors(cudaMemcpy(target, devDt, width * height * sizeof(float), cudaMemcpyDeviceToHost));
}

}

#endif