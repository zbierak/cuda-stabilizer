#include "mFFmpeg/mffmpeg.h"

#include "kanade.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <limits>

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

	int i=0;
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
	//output.WriteVideoFrame((unsigned char*)frame.pFrame->data[0], frame.width, frame.height);

	kanadeNextFrame(frame.pFrame->data[0], frame.width, frame.height);

	unsigned char* result8 = (unsigned char*)malloc(frame.width*frame.height*sizeof(unsigned char));
	unsigned char* result24 = (unsigned char*)malloc(frame.width*frame.height*3*sizeof(unsigned char));

	// przelec przez wszystkie ramki az do konca strumienia
	while ((rfr = input.ReadNextFrame(&frame)) != MFFMpegInput::RFR_STREAMEND)
	{
		if (rfr == MFFMpegInput::RFR_OK)								// was the frame parsed ok?
		{								
			kanadeNextFrame(frame.pFrame->data[0], frame.width, frame.height);
			
			kanadeTestBuildPyramid(result8, frame.width, frame.height);
			kanade8to24(result24, result8, frame.width, frame.height);

				//unsigned char* result = kanadeExecute(frame.width, frame.height);
			output.WriteVideoFrame(result24, frame.width, frame.height);

			cout << ".";
			i++;

			// mozna tez zapisac ramke do pliku ppm poleceniem:
			// frame.SaveAsPPM(fileName);
		}
	}

	free(result8);
	free(result24);


	cout << endl;	

	system("pause");

	kanadeCleanup();

	return 0;
}
