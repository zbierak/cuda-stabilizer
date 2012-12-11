#include <iostream>

std::string IEOUT_Strings[] =  {"no error", "cannot guess file format", "cannot allocate memory", "supplied output format parameters are invalid", "cannot open the output file", "cannot find codec, or other problem with it"};

inline AVStream* MFFMpegOutput::AddVideoStream(AVFormatContext *oc, int codec_id, int bitrate, int framerate, int width, int height)
{
	AVCodecContext *c;
    AVStream *st;

    st = av_new_stream(oc, 0);
    if (!st) return NULL;

    c = st->codec;
    c->codec_id = (CodecID) codec_id;
    c->codec_type = CODEC_TYPE_VIDEO;

    /* put sample parameters */
    c->bit_rate = bitrate;//400000;
    /* resolution must be a multiple of two */
    c->width = width;
    c->height = height;
    /* time base: this is the fundamental unit of time (in seconds) in terms
       of which frame timestamps are represented. for fixed-fps content,
       timebase should be 1/framerate and timestamp increments should be
       identically 1. */
    c->time_base.den = framerate;
    c->time_base.num = 1;
    c->gop_size = 12; /* emit one intra frame every twelve frames at most */
    c->pix_fmt = PIX_FMT_YUV420P;		// zobaczmy, jak to sie bedzie sprawowac...

    if (c->codec_id == CODEC_ID_MPEG2VIDEO) 
	{
        /* just for testing, we also add B frames */
        c->max_b_frames = 2;
    }

    if (c->codec_id == CODEC_ID_MPEG1VIDEO)
	{
        /* Needed to avoid using macroblocks in which some coeffs overflow.
           This does not happen with normal video, it just happens here as
           the motion of the chroma plane does not match the luma plane. */
        c->mb_decision=2;
    }

    // some formats want stream headers to be separate
    if(!strcmp(oc->oformat->name, "mp4") || !strcmp(oc->oformat->name, "mov") || !strcmp(oc->oformat->name, "3gp"))
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;

    return st;
}

inline AVFrame* MFFMpegOutput::AllocPicture(int pix_fmt, int width, int height)
{
    AVFrame* picture;
    uint8_t* picture_buf;
    int size;

    picture = avcodec_alloc_frame();
    if (!picture) return NULL;

    size = avpicture_get_size(pix_fmt, width, height);
    picture_buf = (uint8_t*)av_malloc(size);

    if (!picture_buf) 
	{
        av_free(picture);
        return NULL;
    }

    avpicture_fill((AVPicture *)picture, picture_buf, pix_fmt, width, height);

    return picture;
}

inline bool MFFMpegOutput::OpenVideo(AVFormatContext* oc, AVStream* st)
{
    AVCodec* codec;
    AVCodecContext* c;

    c = st->codec;

    /* find the video encoder */
    codec = avcodec_find_encoder(c->codec_id);
    if (!codec) return false;

    /* open the codec */
    if (avcodec_open(c, codec) < 0) return false;

    video_outbuf = NULL;
    if (!(oc->oformat->flags & AVFMT_RAWPICTURE)) 
	{
        /* allocate output buffer */
        /* XXX: API change will be done */
        /* buffers passed into lav* can be allocated any way you prefer,
           as long as they're aligned enough for the architecture, and
           they're freed appropriately (such as using av_free for buffers
           allocated with av_malloc) */
        video_outbuf_size = 200000;
        video_outbuf = (uint8_t*) av_malloc(video_outbuf_size);
    }

    /* allocate the encoded raw picture */
    picture = AllocPicture(c->pix_fmt, c->width, c->height);
    if (!picture) return false;

    /* if the output format is not YUV420P, then a temporary YUV420P
       picture is needed too. It is then converted to the required
       output format */
    tmp_picture = NULL;
    if (c->pix_fmt != PIX_FMT_RGB24) 
	{
        tmp_picture = AllocPicture(PIX_FMT_RGB24, c->width, c->height);
        if (!tmp_picture) return false;
    }

	return true;
}

inline MFFMpegOutput::InitError MFFMpegOutput::Init(const char* fileName, int bitrate, int framerate, int width, int height)
{
	fmt = guess_format(NULL, fileName, NULL);
    
	if (!fmt) fmt = guess_format("mpeg", NULL, NULL);		// cannot guess format - use mpeg
	if (!fmt) return IE_CANNOT_GUESS_FORMAT;				// cannot guess format at all
        
	// alocate output memory

	oc = av_alloc_format_context();
    if (!oc) return IE_CANNOT_ALLOCATE_MEMORY;
    oc->oformat = fmt;

	audio_st = NULL;
	video_st = NULL;
    if (fmt->video_codec != CODEC_ID_NONE) video_st = AddVideoStream(oc, fmt->video_codec, bitrate, framerate, width, height);
	// in here - allocation of audio buffer - it's not here, as it's not used yet
    
	if (av_set_parameters(oc, NULL) < 0) return IE_INVALID_PARAMETERS;
    
	dump_format(oc, 0, fileName, 1);
    
	if (video_st) if (!OpenVideo(oc, video_st)) return IE_PROBLEMS_WITH_CODEC;
	//if (audio_st) open_audio(oc, audio_st);

	if (!(fmt->flags & AVFMT_NOFILE)) 
        if (url_fopen(&oc->pb, fileName, URL_WRONLY) < 0) return IE_CANNOT_OPEN;

	av_write_header(oc);

	return IE_NONE;
}

inline MFFMpegOutput::~MFFMpegOutput()
{
	// close the codecs
	if (video_st) 
	{
		avcodec_close(video_st->codec);
		av_free(picture->data[0]);
		av_free(picture);
		if (tmp_picture) 
		{
			av_free(tmp_picture->data[0]);
			av_free(tmp_picture);
		}
		av_free(video_outbuf);
	}

    //if (audio_st) close_audio(oc, audio_st);

	av_write_trailer(oc);

	for(unsigned i = 0; i < oc->nb_streams; i++) 
	{
        av_freep(&oc->streams[i]->codec);
        av_freep(&oc->streams[i]);
    }

	if (!(fmt->flags & AVFMT_NOFILE)) url_fclose(oc->pb);
    
	av_free(oc);
}

inline bool MFFMpegOutput::WriteVideoFrame(unsigned char* mem, int width, int height)
{
	int out_size, ret;
    AVCodecContext *c;
    static struct SwsContext *img_convert_ctx;
	
	c = video_st->codec;

	if (c->pix_fmt != PIX_FMT_RGB24) 
	{
		    /* as we only generate a YUV420P picture, we must convert it
               to the codec pixel format if needed */
		if (img_convert_ctx == NULL) 
		{
			img_convert_ctx = sws_getContext(c->width, c->height,
                                             PIX_FMT_RGB24,
                                             c->width, c->height,
                                             c->pix_fmt,
                                             SWS_BICUBIC, NULL, NULL, NULL);
            if (img_convert_ctx == NULL) return false;
        }
        
		//fill_yuv_image(tmp_picture, frame_count, c->width, c->height);

		for(int y=0; y<height; y++)
			memcpy(tmp_picture->data[0]+y*tmp_picture->linesize[0], mem+3*y*width, 3*width);
	
		//tmp_picture->data[0] = mem;

        sws_scale(img_convert_ctx, tmp_picture->data, tmp_picture->linesize,
				  0, c->height, picture->data, picture->linesize);

    } 
	else
	{
	//	picture->data[0] = mem;
		for(int y=0; y<height; y++)
			memcpy(picture->data[0]+y*picture->linesize[0], mem+3*y*width, 3*width);
	}

	if (oc->oformat->flags & AVFMT_RAWPICTURE) 
	{
        /* raw video case. The API will change slightly in the near
           futur for that */
        AVPacket pkt;
        av_init_packet(&pkt);

        pkt.flags |= PKT_FLAG_KEY;
        pkt.stream_index = video_st->index;
        pkt.data= (uint8_t *)picture;
        pkt.size= sizeof(AVPicture);

        ret = av_write_frame(oc, &pkt);
    } 
	else 
	{
        /* encode the image */
        out_size = avcodec_encode_video(c, video_outbuf, video_outbuf_size, picture);
        /* if zero size, it means the image was buffered */
        if (out_size > 0) 
		{
            AVPacket pkt;
            av_init_packet(&pkt);

            if (c->coded_frame->pts != long long int(0x8000000000000000))
                pkt.pts = av_rescale_q(c->coded_frame->pts, c->time_base, video_st->time_base);
            if(c->coded_frame->key_frame)
                pkt.flags |= PKT_FLAG_KEY;
            pkt.stream_index = video_st->index;
            pkt.data = video_outbuf;
            pkt.size = out_size;

            /* write the compressed frame in the media file */
            ret = av_write_frame(oc, &pkt);
        } 
		else 
            ret = 0;
    }

    if (ret != 0) return false;

	return true;
}

inline std::string MFFMpegOutput::GetError(InitError err)
{
	return IEOUT_Strings[err];
}