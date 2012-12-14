#ifndef __MZBIERSK_MFFMPEG_MFRAME
#define __MZBIERSK_MFFMPEG_MFRAME

#define _CRT_SECURE_NO_WARNINGS

extern "C" 
{
	#include "libavcodec/avcodec.h"
	#include "libavformat/avformat.h"
	#include "libswscale/swscale.h"
}

#include <string>

struct MFrame
{
	AVFrame* pFrame;							// this is pointer only - it's allocated in mffmpedinput Init() function, always shows to MFFMpegInput::pFrameRGB variable and can change while calling MFFMpegInput::ReadNextFrame()
	int width;
	int height;

	bool SaveAsPPM(std::string fileName);
	void SetFrame(AVFrame* _pFrame, int _width, int _height) 
	{
		pFrame = _pFrame;
		width = _width;
		height = _height;
	}
};

inline bool SaveAsPPM32(unsigned char* ptr, std::string fileName, int width, int height);
inline bool SaveAsPPM24(unsigned char* ptr, std::string fileName, int width, int height);
inline bool SaveAsPPM8(unsigned char* ptr, std::string fileName, int width, int height);



#include "mframe_imp.h"

#endif