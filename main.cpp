#include "mFFmpeg/mffmpeg.h"

#include "kanade.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <limits>

// possible preprocessor parameters:
//	KANADE_NO_GPU - the CPU code is compiled & executed (no CUDA card required)
//	KANADE_NO_RFRAME_UPDATE - the reference frame is not updated (the first frame is the reference frame)

using namespace std;

int main()
{
	kanadeInit();

	MFFMpegInput input;
	MFFMpegOutput output;

	string file, outputfile;
	//cout << "Input file: ";
	//cin >> file;

	//cout << "Output file: ";
	//cin >> outputfile;

	//file = "c.mov";
	file = "hippo.mkv";
	//file = "MOV00A.MOD";
	//file = "out1.mkv";
	outputfile = "out.mkv";

	MFFMpegInput::InitError ie = input.Init(file.c_str());
	if (ie != MFFMpegInput::IE_NONE)
	{
		cout << "MFFMpegInput error: " << input.GetError(ie) << endl;
		system("pause");
		return 1;
	}

	MFFMpegOutput::InitError ieo = output.Init(outputfile.c_str(), 4000000, 25, input.GetCodecContext()->width, input.GetCodecContext()->height);
	if (ieo != MFFMpegInput::IE_NONE)
	{
		cout << "MFFMpegOutput error: " << output.GetError(ieo) << endl;
		system("pause");
		return 1;
	}

	MFrame frame;
	MFFMpegInput::ReadFrameResult rfr = MFFMpegInput::RFR_IGNORE;

	while (rfr != MFFMpegInput::RFR_OK && rfr != MFFMpegInput::RFR_STREAMEND) 
		rfr = input.ReadNextFrame(&frame);

	if (rfr == MFFMpegInput::RFR_STREAMEND)
	{
		cout << "The stream does not contain any video. Aborting." << endl;
		system("pause");
		return 1;
	}

	// poki co nie pisz pierwszej klatki
	output.WriteVideoFrame((unsigned char*)frame.pFrame->data[0], frame.width, frame.height);

	kanadeNextFrame(frame.pFrame->data[0], frame.width, frame.height);
	//kanadeTestInit(frame.pFrame->data[0], frame.width, frame.height);

	unsigned char* result8 = (unsigned char*)malloc(frame.width*frame.height*sizeof(unsigned char));
	unsigned char* result24 = (unsigned char*)malloc(frame.width*frame.height*3*sizeof(unsigned char));

	// przelec przez wszystkie ramki az do konca strumienia
	while ((rfr = input.ReadNextFrame(&frame)) != MFFMpegInput::RFR_STREAMEND)
	{
		if (rfr == MFFMpegInput::RFR_OK)								// was the frame parsed ok?
		{			
			kanadeNextFrame(frame.pFrame->data[0], frame.width, frame.height);			
			kanadeExecute(result24, frame.width, frame.height);
			
			//kanadeTestNextFrame(frame.pFrame->data[0], frame.width, frame.height);			
			//kanadeTestCompareBuildPyramid(result8, frame.width, frame.height);
			//kanadeTestGenerateG(result8, frame.width, frame.height);			
			//kanadeTestGenerateB(result8, frame.width, frame.height);
			//kanadeTestBuildPyramid(result8, frame.width, frame.height);
			//kanade8to24(result24, result8, frame.width, frame.height);
			//kanadeTestTranslate(result24, 10.2f, 10.2f, frame.width, frame.height);
			//kanadeTestPrepareForNextFrame();

			//system("pause");

			output.WriteVideoFrame(result24, frame.width, frame.height);
			
			cout << ".";

			
		}
	}

	free(result8);
	free(result24);


	cout << endl;	

	kanadeCleanup();
	kanadePrintStats();

	system("pause");

	return 0;
}
