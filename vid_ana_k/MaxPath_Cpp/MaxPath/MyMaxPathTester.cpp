#include "StdAfx.h"
#include "MyMaxPathTester.h"

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/


CMyMaxPathTester::CMyMaxPathTester(void)
{
}

CMyMaxPathTester::~CMyMaxPathTester(void)
{
}

bool CMyMaxPathTester::TestOneVolume(char* fn_sparse_vol, char* fn_settings, char* fn_output, float small_neg)
{
	MY2DBOX *arrBox;
	CMyMaxPath maxpath;
	CMySparseVolume vol;
	FILE* f;
	float best_path_score;
	int i;

	if (!vol.LoadFromFile(fn_sparse_vol))
	{
		printf("Failed to read the volume data file!\n");
		return false;
	}
	if (!maxpath.LoadSettingsFromFile(fn_settings))
	{
		printf("Failed to read the setting parameters from file!\n");
		return false;
	}
	
	arrBox = (MY2DBOX*)malloc(vol.GetLength()*sizeof(MY2DBOX));
	if (arrBox==NULL)
	{
		printf("Not enough memory!\n");
		return false;
	}
	best_path_score = maxpath.MaxPathSearch(&vol, arrBox, small_neg);

	fopen_s(&f, fn_output, "wt");
	fprintf(f,"%f\n", best_path_score);
	for (i=0; i<vol.GetLength(); i++)
		fprintf(f,"%d %d %d %d\n", arrBox[i].left, arrBox[i].top, arrBox[i].right, arrBox[i].bottom);
	fclose(f);

	free(arrBox);
	return true;
}