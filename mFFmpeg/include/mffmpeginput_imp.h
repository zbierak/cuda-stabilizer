std::string IE_Strings[] =  {"no error", "file doesn't exist", "no stream info in given file", "no apropriate codec found", "cannot open codec", "no free memory left"};

// Function that inits reading from video file
inline MFFMpegInput::InitError MFFMpegInput::Init(const char* fileName)
{
	// Register all formats, codecs and custom log callback
    av_register_all();
	av_log_set_callback(&(MFFMpegInput::LogFunction));

	// Try opening video file
	if (av_open_input_file(&pFormatCtx, fileName, NULL, 0, NULL) != 0) return IE_FILENOTEXISTS; 

	// Retrieve stream information
    if (av_find_stream_info(pFormatCtx) < 0) return IE_NOSTREAMINFO; 

	// Dump information about file onto standard error
    dump_format(pFormatCtx, 0, fileName, false);

	// Find video stream
	videoStream = -1;
	for (unsigned i=0; i<pFormatCtx->nb_streams; i++)
        if(pFormatCtx->streams[i]->codec->codec_type==CODEC_TYPE_VIDEO)
        {
            videoStream=i;
            break;
        }

    if (videoStream == -1) return IE_NOVIDEOSTREAM;						// Didn't find a video stream

	// Get a pointer to the codec context for the video stream
    pCodecCtx = pFormatCtx->streams[videoStream]->codec;

	// Find the decoder for the video stream
	pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
    if (pCodec == NULL) return IE_NOCODECFOUND;							// Codec not found

	// Try opening codec
    if (avcodec_open(pCodecCtx, pCodec) < 0) return IE_CANTOPENCODEC;	// Could not open codec

	// Allocate video frame
    pFrame = avcodec_alloc_frame();
	pFrameRGB = avcodec_alloc_frame();
    if (pFrameRGB == NULL) return IE_NOMEMORYLEFT;

	// Determine required buffer size and allocate buffer
    numBytes = avpicture_get_size(PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);
    buffer = new uint8_t[numBytes];

	// Assign appropriate parts of buffer to image planes in pFrameRGB
    avpicture_fill((AVPicture *)pFrameRGB, buffer, PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);

	return IE_NONE;
}

// Tidying up after reading a video file
inline MFFMpegInput::~MFFMpegInput()
{
	// Free the RGB image
    delete[] buffer;
    av_free(pFrameRGB);

    // Free the YUV frame
    av_free(pFrame);

    // Close the codec
    avcodec_close(pCodecCtx);

    // Close the video file
    av_close_input_file(pFormatCtx);
}


inline MFFMpegInput::ReadFrameResult MFFMpegInput::ReadNextFrame(MFrame* resFrame)
{
	if (av_read_frame(pFormatCtx, &packet) < 0) return RFR_STREAMEND;	// no more frames to read

	// Is this a packet from the video stream?
    if (packet.stream_index != videoStream)								// something's wrong - should never happen (hope so :)
	{
		av_free_packet(&packet);
		return RFR_IGNORE;				
	}

	// Decode the frame
	int frameFinished;
	do
	{
		// Decode video frame
		avcodec_decode_video(pCodecCtx, pFrame, &frameFinished, packet.data, packet.size);
		if (!frameFinished)												// frame's not finished - decode next frame (hope so :)
		{
			av_free_packet(&packet);
			if (av_read_frame(pFormatCtx, &packet) < 0) return RFR_IGNORE;
		}
	} while (!frameFinished);
      
	// Convert the image from its native format to RGB

 
    SwsContext* img_convert_ctx = sws_getContext(pCodecCtx->width, 
												 pCodecCtx->height, 
												 pCodecCtx->pix_fmt, 
												 pCodecCtx->width, 
												 pCodecCtx->height,
												 PIX_FMT_RGB24,
												 SWS_FAST_BILINEAR,		// sws_flags
												 NULL, NULL, NULL);
	if (img_convert_ctx == NULL) return RFR_IGNORE;						// something's wrong w/h image convertion, prevent from crashing								
	sws_scale(img_convert_ctx, pFrame->data, pFrame->linesize, 0, pCodecCtx->height, pFrameRGB->data, pFrameRGB->linesize);
	sws_freeContext(img_convert_ctx);

	// free the packet that was allocated by av_read_frame
    av_free_packet(&packet);

	// return encoded frame

	resFrame->SetFrame(pFrameRGB, pCodecCtx->width, pCodecCtx->height);
	return RFR_OK;
}

inline std::string MFFMpegInput::GetError(InitError err)
{
	return IE_Strings[err];
}