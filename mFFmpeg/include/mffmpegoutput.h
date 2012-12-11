#ifndef __MZBIERSK_MFFMPEG_MFFMPEGOUTPUT
#define __MZBIERSK_MFFMPEG_MFFMPEGOUTPUT

extern "C" 
{
	#include "libavformat/avformat.h"
	#include "libavcodec/avcodec.h"
	#include "libswscale/swscale.h"
}

#include <string>

class MFFMpegOutput
{
public:
	enum InitError { IE_NONE, IE_CANNOT_GUESS_FORMAT, IE_CANNOT_ALLOCATE_MEMORY, IE_INVALID_PARAMETERS, IE_CANNOT_OPEN, IE_PROBLEMS_WITH_CODEC};
private:
	AVOutputFormat* fmt;
    AVFormatContext* oc;
    AVStream* video_st, *audio_st;
    double video_pts, audio_pts;

	AVFrame *picture, *tmp_picture;
	uint8_t *video_outbuf;
	int frame_count, video_outbuf_size;

	AVStream* AddVideoStream(AVFormatContext *oc, int codec_id, int bitrate, int framerate, int width, int height);
	AVFrame* AllocPicture(int pix_fmt, int width, int height);
	bool OpenVideo(AVFormatContext *oc, AVStream *st);
public:
	InitError Init(const char* fileName, int bitrate, int framerate, int width, int height);
	~MFFMpegOutput();
	bool WriteVideoFrame(unsigned char* mem, int width, int height); 	
	std::string GetError(InitError err);
};

#include "mffmpegoutput_imp.h"

#endif