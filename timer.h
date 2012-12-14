/* class for measurint the amount of elapsed time (that said, a timer :)
 * author: mzbiersk, based on the tick_count class from Intel TBB */

// On RedHat and other linux system you might need to add "-lrt" to your compile line
// or perhaps not :)

#ifndef __MZBIERSK_TIMER
#define __MZBIERSK_TIMER

#if _WIN32||_WIN64
	#include <windows.h>
	#define TIME_TYPE LARGE_INTEGER 
#else 
 	#include <sys/time.h>
	#define TIME_TYPE struct timeval
#endif 

struct UniversalTime
{	
	static TIME_TYPE getCurrentTime()
	{
		TIME_TYPE startTime;

		#if _WIN32||_WIN64
			QueryPerformanceCounter(&startTime);
		#else 
			gettimeofday(&startTime, NULL);
		#endif

		return startTime;
	}	
};

class Timer
{
private:
	TIME_TYPE startTime;
	TIME_TYPE stopTime;
	
	#if _WIN32||_WIN64
		double frequency;
	#endif
public:
	Timer()
	{
		#if _WIN32||_WIN64
			LARGE_INTEGER proc_freq;
			QueryPerformanceFrequency(&proc_freq);
			frequency = (double)proc_freq.QuadPart;
		#endif

		//UniversalTime::getCurrentTime();
		restart();
	}

	Timer(TIME_TYPE _startTime)
	{
		#if _WIN32||_WIN64
			LARGE_INTEGER proc_freq;
			QueryPerformanceFrequency(&proc_freq);
			frequency = (double)proc_freq.QuadPart;
		#endif		

		startTime = _startTime;	
	}

	void restart()
	{
		#if _WIN32||_WIN64
			QueryPerformanceCounter(&startTime);
		#else 
			gettimeofday(&startTime, NULL);
		#endif
	}

	double stop()
	{
		#if _WIN32||_WIN64
			QueryPerformanceCounter(&stopTime);
			return ((stopTime.QuadPart - startTime.QuadPart) / frequency);
		#else 
			gettimeofday(&stopTime, NULL);
			return (stopTime.tv_sec - startTime.tv_sec) + (stopTime.tv_usec - startTime.tv_usec) / 1000000.0;
		#endif
	}

};
#endif