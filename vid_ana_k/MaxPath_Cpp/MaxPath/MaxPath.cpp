// MaxPath.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "MaxPath.h"
#include "MyMaxPathTester.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// The one and only application object
bool TestOneVolume(char* fn_vol, char* fn_settings, char* fn_out, float small_neg);

CWinApp theApp;

using namespace std;

int _tmain(int argc, TCHAR* argv[], TCHAR* envp[])
{
	int nRetCode = 0;

	// initialize MFC and print and error on failure
	if (!AfxWinInit(::GetModuleHandle(NULL), NULL, ::GetCommandLine(), 0))
	{
		// TODO: change error code to suit your needs
		_tprintf(_T("Fatal Error: MFC initialization failed\n"));
		nRetCode = 1;
	}
	else
	{
		// TODO: code your application's behavior here.
		if (argc != 5)
		{
			printf("Usage MaxPath fn_volume fn_settings fn_output small_negative_value\n");
		}
		else
		{
			if (TestOneVolume(CW2A(argv[1]), CW2A(argv[2]), CW2A(argv[3]), float(atof(CW2A(argv[4])))))
			{
				printf("Finished searching, output is in %s\n", CW2A(argv[3]));
			}
			else
			{
				printf("Data format failed!");
			}
		}
	}

	return nRetCode;
}
bool TestOneVolume(char* fn_vol, char* fn_settings, char* fn_out, float small_neg)
{
	CMyMaxPathTester tester;
	return tester.TestOneVolume(fn_vol, fn_settings, fn_out, small_neg);
}