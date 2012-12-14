#include <stdio.h>

inline bool MFrame::SaveAsPPM(std::string fileName)
{
    FILE* pFile;

    // Open file
	pFile = fopen(fileName.c_str(), "wb");
    if (pFile == NULL) return false;

    // Write header
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    for(int y=0; y<height; y++)
        fwrite(pFrame->data[0]+y*pFrame->linesize[0], 1, width*3, pFile);

    // Close file
    fclose(pFile);

	return true;
}

// zapisz jako ppm redukujac co 4ty bajt
inline bool SaveAsPPM32(unsigned char* ptr, std::string fileName, int width, int height)
{
    FILE* pFile;

    // Open file
	pFile = fopen(fileName.c_str(), "wb");
    if (pFile == NULL) return false;

    // Write header
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
	long size = 4*width*height;
    for(long i=0; i<size; i++)
	{
		if (i%4 != 3)
			fwrite(ptr+i, 1, 1, pFile);
	}

    // Close file
    fclose(pFile);

	return true;
}

inline bool SaveAsPPM24(unsigned char* ptr, std::string fileName, int width, int height)
{
    FILE* pFile;

    // Open file
	pFile = fopen(fileName.c_str(), "wb");
    if (pFile == NULL) return false;

    // Write header
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
	fwrite(ptr, 1, 3*width*height, pFile);

    // Close file
    fclose(pFile);

	return true;
}

// zapisz jako ppm rozszerzajac grejskale na 24b
inline bool SaveAsPPM8(unsigned char* ptr, std::string fileName, int width, int height)
{
    FILE* pFile;

    // Open file
	pFile = fopen(fileName.c_str(), "wb");
    if (pFile == NULL) return false;

    // Write header
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
	long size = width*height;
    for(long i=0; i<size; i++)
	{
		fwrite(ptr+i, 1, 1, pFile);
		fwrite(ptr+i, 1, 1, pFile);
		fwrite(ptr+i, 1, 1, pFile);
	}

    // Close file
    fclose(pFile);

	return true;
}