#ifndef __MZBIERSK_MFFMPEG_MFFMPEGINPUT
#define __MZBIERSK_MFFMPEG_MFFMPEGINPUT

#include "mframe.h"

extern "C" 
{
	#include "libavcodec/avcodec.h"
	#include "libavformat/avformat.h"
	#include "libswscale/swscale.h"
}

#include <string>

class MFFMpegInput
{
public:
	enum InitError { IE_NONE, IE_FILENOTEXISTS, IE_NOSTREAMINFO, IE_NOVIDEOSTREAM, IE_NOCODECFOUND, IE_CANTOPENCODEC, IE_NOMEMORYLEFT};
	enum ReadFrameResult { RFR_OK, RFR_STREAMEND, RFR_IGNORE };
private:
	AVFormatContext *pFormatCtx;
	AVCodecContext* pCodecCtx;
	AVCodec* pCodec;
	AVFrame* pFrame; 
	AVFrame* pFrameRGB;
	AVPacket packet;
	int numBytes;
    uint8_t* buffer;
	int videoStream;
protected:
	static void LogFunction(void* ptr, int level, const char* fmt, va_list vl) {}					// logging code
public:
	InitError Init(const char* fileName);
	~MFFMpegInput();
	ReadFrameResult ReadNextFrame(MFrame* resFrame);	
	std::string GetError(InitError err);
	AVCodecContext* GetCodecContext() { return pCodecCtx; }
	int GetVideoWidth() { return pCodecCtx->width; }
	int GetVideoHeight() { return pCodecCtx->height; }
};

#include "mffmpeginput_imp.h"

#endif