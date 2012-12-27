#ifndef KANADE_NO_GPU

namespace kgpu {

const unsigned BLOCK_DIM = 32;

#include "kanade.h"
#include "helper_cuda.h"

unsigned char* ioFrame24 = NULL;									// pamiec pod ramke 24b - uzywana na wejsciu i wyjsciu
unsigned char* ioFrame32 = NULL;									// pamiec pod ramke 32b - uzywana na wejsciu jako tymczasowa dla tekstury
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

texture<uchar4, 2, cudaReadModeElementType> frameTex;				// kolorowa ramka wejœciowa do przesuwania	
cudaArray* frameTexMem = NULL;										

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

__global__ void prepareFrame(unsigned char* frame24, unsigned char* frame32, unsigned char* frame8, unsigned width, unsigned height)
{
	// @todo: cala te funkcje mozna zrobic wydajniej przy uzyciu pamieci dzielonej

	int pos = blockIdx.x*blockDim.x + threadIdx.x;
	if (pos >= width*height) return;

	unsigned char r = frame24[3*pos];
	unsigned char g = frame24[3*pos+1];
	unsigned char b = frame24[3*pos+2];

	// konwersja na skale szarosci wg. jasnosci piksela
	//frame8[pos] = toColor(floor(0.299f * r + 0.587 * g + 0.114 * b));		// ten sposob (przeksztalcenie na YUV) powodowal bledy miedzy implementacja CPU a GPU
	//frame8[pos] = toColor(floor((r + g + b)/3.0f));
	frame8[pos] = MAX(MAX(r, g), b);
	
	// konwersja formatu 24b na 32b dla kompatybilnosci z teksturami (wartosc alfa jest zerowana przed wywolaniem)
	if (frame32 != NULL)
	{
		frame32[4*pos] = r;
		frame32[4*pos+1] = g;
		frame32[4*pos+2] = b;
	}
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

	// werja nearest neighbour
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

	// wersja interpolujaca
/*
	int dx = (int) floor(vx);					// @todo: te obliczenia mozna sprobowac wykonac na cpu i wrzucic do pamieci stalej
	int dy = (int) floor(vy);

	double nx = vx - floor(vx);					// nx - fraction of the next pixel in x taken into interpolation
	double ny = vy - floor(vy);
	double tx = 1 - nx;							// tx - fraction of thix pixel taken into interpolation
	double ty = 1 - ny;

	unsigned dpos = 3*(y * width + x);

	if (x-dx >= 0 && y-dy >= 0 && x-dx < width-1 && y-dy < height-1)
	{
		unsigned spos = 3*((y - dy) * width + x - dx);
		for (unsigned c=0; c<3; c++) 
		{
			float y1 = (tx * input24[spos+c] + nx * input24[spos+c+3]);
			float y2 = (tx * input24[spos+c+3*width] + nx * input24[spos+c+3*width+3]);

			output24[dpos+c] = (unsigned char)(ty * y1 + ny * y2);
		}
	}
*/
	//uchar4 c = tex2D(frameTex, x - vx, y - vy);

	// @todo: to mozna zrobic wydajniej przy uzyciu pamieci dzielonej
	//int pos = 3*(y*width + x);
	//output24[pos] = c.x;
	//output24[pos+1] =  c.y;
	//output24[pos+2] =  c.z;
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
	else
		dx[pos] = 0;

	if (y != 0 && y != height-1)
		dy[pos] = ((float)(frame8[pos+width]-frame8[pos-width])) / 2.0f;
	else
		dy[pos] = 0;	
}

__global__ void calculate_dt(unsigned pyrLvl, float vx, float vy, unsigned char* prevFrame8b, unsigned char* currFrame8b, float* dt, unsigned width, unsigned height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	// zezwol tylko na wartosci z poprawnego zakresu
	if (x >= width || y >= height) return;

	int dx = (int) floor(vx);
	int dy = (int) floor(vy);

	double nx = vx - floor(vx);					// fractions of the next pixel taken into interpolation
	double ny = vy - floor(vy);
	double tx = 1 - nx;							// fraction of this pixel taken into interpolation
	double ty = 1 - ny;
	
	float interpolated = 0;
	if (x+dx >= 0 && y+dy >= 0 && x+dx < width-1 && y+dy < height-1)
	{
		unsigned spos = (y + dy) * width + x + dx;
		interpolated = (unsigned char)((tx * currFrame8b[spos] + nx * currFrame8b[spos + 1] + ty * currFrame8b[spos] + ny * currFrame8b[spos + width]) / 2.0f);		// na 2 a nie na 4, bo korzystamy z "polkolorow" (w punkcie srodkowym kazdy kolor jest mnozony przez 0.5)
	}

	int dpos = y*width + x;
	dt[dpos] = (float)prevFrame8b[dpos] - interpolated;
}

// for full image operations
template <typename T1, typename T2>
__device__ __host__ float dotSum(T1* a, T2* b, int size)
{
	float sum = 0.0f;
	for (int i=0; i<size; i++)
		sum += a[i] * b[i];
	return sum;
}

// for window operations
template <typename T1, typename T2>
__device__ __host__ float dotSum(T1* a, T2* b, int width, int height)
{
	float sum = 0.0f;
	int px = width/2;
	int py = height/2;

	int sx = MAX(0, px-(int)WINDOW_SIZE_DIV_2);
	int ex = MIN(width-1, px+WINDOW_SIZE_DIV_2);
	int sy = MAX(0, py-(int)WINDOW_SIZE_DIV_2);
	int ey = MIN(height-1, py+WINDOW_SIZE_DIV_2);

	for (int y=sy; y<=ey; y++)
		for (int x=sx; x<=ex; x++)
		{
			int pos = y*width+x;
			sum += a[pos] * b[pos];
		}
	
	return sum;
}

// reduce to g

/*
__device__ unsigned int kernel_block_counter = 0;
__shared__ bool kernel_last_block_done;

__device__ float g_partial_sum(float* dx, float* dy)
{
	__shared__ float t_dxdx[BLOCK_DIM];
	__shared__ float t_dxdy[BLOCK_DIM];
	__shared__ float t_dydy[BLOCK_DIM];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	float l_dx = dx[idx];
	float l_dy = dy[idx];

	t_dxdx[threadIdx.x] = l_dx * l_dx;
	t_dxdy[threadIdx.x] = l_dx * l_dy;
	t_dydy[threadIdx.x] = l_dy * l_dy;

	__syncthreads();

	int nTotalThreads = blockDim.x;					 
	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two		
 
		if (threadIdx.x < halfPoint)
		    t_dxdx[threadIdx.x] += t_dxdx[threadIdx.x + halfPoint];
 
		__syncthreads();
 
		nTotalThreads = halfPoint;
	}

	return t_dxdx[0];
}

__device__ float g_total_sum(float* dxdx, float* dxdy, float* dydy)
{
	float sum = 0;
	for (int i=0; i<gridDim.x; i++)
		sum += dxdx[i];
	return sum;
}

__global__ void reduce_to_g(float* dx, float* dy, float* dxdx, float* dxdy, float* dydy, unsigned size)
{
	float partialSum = g_partial_sum(dx, dy);
	if (threadIdx.x == 0)
	{
		dxdx[blockIdx.x] = partialSum;
		__threadfence();
		unsigned int value = atomicInc(&kernel_block_counter, gridDim.x);
		kernel_last_block_done = (value == (gridDim.x - 1));
	}

	__syncthreads();

	if (kernel_last_block_done) 
	{
		float totalSum = g_total_sum(dxdx, dxdy, dydy);
		if (threadIdx.x == 0) 
		{
			dxdx[0] = totalSum;
			kernel_block_counter = 0;
		}
	}

	
}
*/

// ta wersja wprowadzala znaczaco duze bledy
__global__ void reduce_to_g(float* dx, float* dy, float* dxdx, float* dxdy, float* dydy, unsigned size)
{
	// to tez nie dziala
	/*int idx = threadIdx.x + blockIdx.x * blockDim.x;

	float s1 = dx[idx]*dx[idx];
	float s2 = dx[idx]*dy[idx];
	float s3 = dy[idx]*dy[idx];
	atomicAdd(dxdx,s1);
	atomicAdd(dxdy,s2);
	atomicAdd(dydy,s3);
	*/
	
	__shared__ float t_dxdx[BLOCK_DIM];
	__shared__ float t_dxdy[BLOCK_DIM];
	__shared__ float t_dydy[BLOCK_DIM];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	float l_dx = dx[idx];
	float l_dy = dy[idx];

	t_dxdx[threadIdx.x] = l_dx * l_dx;
	t_dxdy[threadIdx.x] = l_dx * l_dy;
	t_dydy[threadIdx.x] = l_dy * l_dy;

	__syncthreads();

	if (threadIdx.x == 0) 
	{
		float s_dxdx = 0, s_dxdy = 0, s_dydy = 0;

		#pragma unroll
		for (int i = 0; i < BLOCK_DIM; i++)
		{
			s_dxdx += t_dxdx[i];
			s_dxdy += t_dxdy[i];
			s_dydy += t_dydy[i];
		}

		//printf("%f\n", s_dxdx);

		atomicAdd(dxdx, s_dxdx);
		atomicAdd(dxdy, s_dxdy);
		atomicAdd(dydy, s_dydy);
	}
	
}

// na chwile obecna nie wykorzystywane, bo strasznie mulilo - jesli to robic to jakos zmyslniej
__global__ void reduce_to_b(float* dx, float* dy, float* dt, float* bx, float* by)
{
	__shared__ float t_dxdt[BLOCK_DIM];
	__shared__ float t_dydt[BLOCK_DIM];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	float l_dt = dt[idx];

	t_dxdt[threadIdx.x] = dx[idx] * l_dt;
	t_dydt[threadIdx.x] = dy[idx] * l_dt;

	__syncthreads();

	if (threadIdx.x == 0) 
	{
		float s_dxdt = 0, s_dydt = 0;

		#pragma unroll
		for (int i = 0; i < BLOCK_DIM; i++)
		{
			s_dxdt += t_dxdt[i];
			s_dydt += t_dydt[i];
		}

		atomicAdd(bx, s_dxdt);
		atomicAdd(by, s_dydt);
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
		
	if (ioFrame32 == NULL)
		checkCudaErrors(cudaMalloc(&ioFrame32, 4 * width * height * sizeof(unsigned char))); 

	if (gpuFrame == NULL)
		checkCudaErrors(cudaMalloc(&gpuFrame, 3 * width * height * sizeof(unsigned char))); 

	unsigned wthAligned = width * height;
	if (wthAligned % BLOCK_DIM != 0)
		wthAligned += (BLOCK_DIM - wthAligned % BLOCK_DIM);

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

	if (frameTexMem != NULL) 
	{
		checkCudaErrors(cudaUnbindTexture(frameTex));
		cudaFreeArray(frameTexMem);
	}

	// przygotuj dane wejsciowe pod tekstury
	checkCudaErrors(cudaMemcpy(ioFrame24, pixels, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(ioFrame32, 0, 4 * width * height * sizeof(unsigned char)));
	prepareFrame<<<(width*height + BLOCK_DIM-1)/BLOCK_DIM, BLOCK_DIM>>>(ioFrame24, ioFrame32, ioFrame8[0], width, height);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(gpuFrame, ioFrame24, 3 * width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

	// przygotuj tekstury
	/*cudaChannelFormatDesc desc24b = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); 	 
	checkCudaErrors(cudaMallocArray(&frameTexMem, &desc24b, width, height)); 
	checkCudaErrors(cudaDeviceSynchronize());	
	checkCudaErrors(cudaMemcpyToArray(frameTexMem, 0, 0, ioFrame32, 4 * width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice)); 
	checkCudaErrors(cudaBindTextureToArray(frameTex, frameTexMem, desc24b)); */
	
}

void kanadePrepareForNextFrame(unsigned width[PYRAMID_SIZE], unsigned height[PYRAMID_SIZE])
{
	/*
	 * right now we have the translated frame in ioFrame24 (as this function is called at the end of kanadeExecute())
	 * Consequently, we will use the translated frame as a reference frame for the next frame. Hence, we have to 
	 * build the pyramid (again) for the translated frame, duh. 
	 */

	// prepare pyramid level zero
	prepareFrame<<<(width[0]*height[0] + BLOCK_DIM-1)/BLOCK_DIM, BLOCK_DIM>>>(ioFrame24, NULL, prevFrame8[0], width[0], height[0]);
	checkCudaErrors(cudaDeviceSynchronize());

	// prepare other pyramid levels
	for (unsigned i=1; i<PYRAMID_SIZE; i++)
	{
		dim3 dimGrid((width[i] + BLOCK_DIM - 1) / BLOCK_DIM, (height[i] + BLOCK_DIM - 1) / BLOCK_DIM);	
		dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	

		build_pyramid_level<<<dimGrid, dimBlock>>>(prevFrame8[i-1], prevFrame8[i], width[i], height[i]);

		// musimy sie synchronizowac, bo kolejne poziomy wymagaja obliczen z popzednich
		checkCudaErrors(cudaDeviceSynchronize());
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
	// potrzebujemy wyrownac do BLOCK_DIM, zeby jadro redukcji dzialalo poprawnie
	unsigned sizeAligned = width * height;
	if (sizeAligned % BLOCK_DIM != 0)
		sizeAligned += (BLOCK_DIM - sizeAligned % BLOCK_DIM);
	checkCudaErrors(cudaMemset(devDx, 0, sizeAligned * sizeof(float)));
	checkCudaErrors(cudaMemset(devDy, 0, sizeAligned * sizeof(float)));

	dim3 dimGrid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	
	calculate_dxdy<<<dimGrid, dimBlock>>>(prevFrame8[pyrLvl], devDx, devDy, width, height);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemset(devG, 0, 3 * sizeof(float)));
	reduce_to_g<<<sizeAligned / BLOCK_DIM, BLOCK_DIM>>>(devDx, devDy, &devG[0], &devG[1], &devG[2], sizeAligned);
	checkCudaErrors(cudaGetLastError());

	//reduce_to_g<<<(width * height + REDUCTION_BLOCK_DIM - 1) / REDUCTION_BLOCK_DIM, REDUCTION_BLOCK_DIM>>>(devDx, devDy, &devG[0], NULL, NULL, sizeAligned);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(cpuG, devG, 3 * sizeof(float), cudaMemcpyDeviceToHost));
	dxdx = cpuG[0];
	dxdy = cpuG[1];
	dydy = cpuG[2];

	
	/*
	// to poki co musi byc kopiowane dla obliczania wektora B
	unsigned size = width * height;
	checkCudaErrors(cudaMemcpy(cpuDx, devDx, size * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cpuDy, devDy, size * sizeof(float), cudaMemcpyDeviceToHost));

	dxdx = dotSum(cpuDx, cpuDx, size);
	dxdy = dotSum(cpuDx, cpuDy, size);
	dydy = dotSum(cpuDy, cpuDy, size);
	//dxdx = dotSum(cpuDx, cpuDx, width, height);
	//dxdy = dotSum(cpuDx, cpuDy, width, height);
	//dydy = dotSum(cpuDy, cpuDy, width, height);
	*/
}

void kanadeCalculateB(unsigned pyrLvl, float vx, float vy, unsigned width, unsigned height, float& bx, float& by)
{
	unsigned sizeAligned = width * height;
	if (sizeAligned % BLOCK_DIM != 0)
		sizeAligned += (BLOCK_DIM - sizeAligned % BLOCK_DIM);
	checkCudaErrors(cudaMemset(devDt, 0, sizeAligned * sizeof(float)));

	dim3 dimGrid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	
	calculate_dt<<<dimGrid, dimBlock>>>(pyrLvl, vx, vy, prevFrame8[pyrLvl], ioFrame8[pyrLvl], devDt, width, height);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemset(devB, 0, 2 * sizeof(float)));
	reduce_to_b<<<sizeAligned / BLOCK_DIM, BLOCK_DIM>>>(devDx, devDy, devDt, &devB[0], &devB[1]);

	checkCudaErrors(cudaMemcpy(cpuB, devB, 2 * sizeof(float), cudaMemcpyDeviceToHost));
	bx = cpuB[0];
	by = cpuB[1];

	/*
	unsigned size = width * height;
	checkCudaErrors(cudaMemcpy(cpuDt, devDt, size * sizeof(float), cudaMemcpyDeviceToHost));

	bx = dotSum(cpuDx, cpuDt, size);
	by = dotSum(cpuDy, cpuDt, size);
	//bx = dotSum(cpuDx, cpuDt, width, height);
	//by = dotSum(cpuDy, cpuDt, width, height);
	*/
}

// level 0 is automatically initiated elsewhere
void kanadeBuildPyramidLevel(unsigned levelId, unsigned newWidth, unsigned newHeight)
{
	dim3 dimGrid((newWidth + BLOCK_DIM - 1) / BLOCK_DIM, (newHeight + BLOCK_DIM - 1) / BLOCK_DIM);	
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	

	build_pyramid_level<<<dimGrid, dimBlock>>>(ioFrame8[levelId-1], ioFrame8[levelId], newWidth, newHeight);

	// musimy sie synchronizowac, bo kolejne poziomy wymagaja obliczen z popzednich
	checkCudaErrors(cudaDeviceSynchronize());
}

void kanadeInit()
{
	cudaDeviceProp deviceProp;
	int devID = 0;
	checkCudaErrors(cudaSetDevice(devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);	

	// jesli tekstura odwoluje sie poza swoj zakres to zwracany jest kolor zerowy
	frameTex.addressMode[0] = cudaAddressModeBorder;	
    frameTex.addressMode[1] = cudaAddressModeBorder; 

	checkCudaErrors(cudaMalloc(&devG, 3 * sizeof(float))); 
	checkCudaErrors(cudaMalloc(&devB, 2 * sizeof(float))); 
	
	cpuG = (float*) malloc(3 * sizeof(float));
	cpuB = (float*) malloc(2 * sizeof(float));	
}

void kanadeCleanup()
{
	if (frameTexMem != NULL)
		cudaFreeArray(frameTexMem);

	for (unsigned i=0; i<PYRAMID_SIZE; i++)
	{
		if (ioFrame8[i] != NULL)
			cudaFree(ioFrame8[i]);
		if (prevFrame8[i] != NULL)
			cudaFree(prevFrame8[i]);
	}

	if (ioFrame24 != NULL)
		cudaFree(ioFrame24);

	if (ioFrame32 != NULL)
		cudaFree(ioFrame32);

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