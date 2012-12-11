#ifndef KANADE_CPU

const unsigned BLOCK_DIM = 32;

#include "kanade.h"
#include "helper_cuda.h"

unsigned char* cpuMem = NULL;										// pamiec CPU w ktorej przechowujemy ramke wynikowa

unsigned char* ioFrame24 = NULL;									// pamiec pod ramke 24b - uzywana na wejsciu i wyjsciu
unsigned char* ioFrame32 = NULL;									// pamiec pod ramke 32b - uzywana na wejsciu jako tymczasowa dla tekstury
unsigned char* ioFrame8 = NULL;										// pamiec pod ramke 8b  - uzywana na wejsciu jako tymczasowa dla tekstury

unsigned char* gframe;												// ramki w skali szarosci do algorytmu L-K
unsigned char* gref;

texture<uchar4, 2, cudaReadModeElementType> frameTex;				// kolorowa ramka wejœciowa do przesuwania	
cudaArray* frameTexMem = NULL;										

template <typename T>
__device__ inline unsigned char toColor(T value)
{
	if (value > 255) return 255;
	if (value < 0) return 0;
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
	frame8[pos] = toColor(0.299f * r + 0.587 * g + 0.114 * b);
	
	// konwersja formatu 24b na 32b dla kompatybilnosci z teksturami (wartosc alfa jest zerowana przed wywolaniem)
	frame32[4*pos] = r;
	frame32[4*pos+1] = g;
	frame32[4*pos+2] = b;
}

__global__ void translate(float vx, float vy, unsigned char* frame24, unsigned width, unsigned height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	// zezwol tylko na wartosci z poprawnego zakresu
	if (x >= width || y >= height) return;

	uchar4 c = tex2D(frameTex, x - vx, y - vy);

	// @todo: to mozna zrobic wydajniej przy uzyciu pamieci dzielonej
	int pos = 3*(y*width + x);
	frame24[pos] = c.x;
	frame24[pos+1] =  c.y;
	frame24[pos+2] =  c.z;
}

__global__ void gKernel(unsigned char* frame32, float* dx, float* dy, unsigned width, unsigned height)
{
}

void beforeFirstFrame(unsigned width, unsigned height)
{
	if (ioFrame24 == NULL)
		checkCudaErrors(cudaMalloc(&ioFrame24, 3 * width * height * sizeof(unsigned char))); 

	if (ioFrame8 == NULL)
		checkCudaErrors(cudaMalloc(&ioFrame8, width * height * sizeof(unsigned char))); 
		
	if (ioFrame32 == NULL)
		checkCudaErrors(cudaMalloc(&ioFrame32, 4 * width * height * sizeof(unsigned char))); 

	if (cpuMem == NULL)
		cpuMem = (unsigned char*) malloc(3 * width * height * sizeof(unsigned char));
}

void kanadeNextFrame(unsigned char* pixels, unsigned width, unsigned height)
{
	beforeFirstFrame(width, height);

	if (frameTexMem != NULL) 
	{
		checkCudaErrors(cudaUnbindTexture(frameTex));
		cudaFreeArray(frameTexMem);
	}

	// przygotuj dane wejsciowe pod tekstury
	checkCudaErrors(cudaMemcpy(ioFrame24, pixels, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(ioFrame32, 0, 4 * width * height * sizeof(unsigned char)));
	prepareFrame<<<(width*height + BLOCK_DIM-1)/BLOCK_DIM, BLOCK_DIM>>>(ioFrame24, ioFrame32, ioFrame8, width, height);
	checkCudaErrors(cudaGetLastError());

	// przygotuj tekstury
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); 	 
	checkCudaErrors(cudaMallocArray(&frameTexMem, &channelDesc, width, height)); 
	checkCudaErrors(cudaDeviceSynchronize());	
	checkCudaErrors(cudaMemcpyToArray(frameTexMem, 0, 0, ioFrame32, 4 * width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice)); 
	checkCudaErrors(cudaBindTextureToArray(frameTex, frameTexMem, channelDesc)); 
}

unsigned char* kanadeTranslate(unsigned width, unsigned height)
{	
	dim3 dimGrid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);	

	cudaMemset(ioFrame24, 0, 3*width*height*sizeof(unsigned char));
	translate<<<dimGrid, dimBlock>>>(20.5, 30.7, ioFrame24, width, height);
	checkCudaErrors(cudaMemcpy(cpuMem, ioFrame24, 3 * width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	return cpuMem;
}

void calculateG(unsigned width, unsigned height, float& dxdx, float& dxdy, float& dydy)
{

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
}

void kanadeCleanup()
{
	if (frameTexMem != NULL)
		cudaFreeArray(frameTexMem);

	if (ioFrame8 != NULL)
		cudaFree(ioFrame8);

	if (ioFrame24 != NULL)
		cudaFree(ioFrame24);

	if (ioFrame32 != NULL)
		cudaFree(ioFrame32);

	if (cpuMem != NULL)
		free(cpuMem);
}

#endif