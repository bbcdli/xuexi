#include "StdAfx.h"
#include "MyMaxPath.h"
#include <stdlib.h>
#include "MyIntegralImage.h"

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/


CMyMaxPath::CMyMaxPath(void)
{
	pPointList = NULL;
	// default values
	nNumOfScales = NUM_OF_SCALES;
	nScaleStep = SCALE_STEP;
	nSpacialStep = SPACIAL_STEP;
	nStartScale = START_SCALE;
	nRadius = NRADIUS;
	nNumOfRatios = NUM_OF_RATIOS;
	fSmallestRatio = float(SMALLEST_RATIO);
	fRatioStep = float(RATIO_STEP);
}
void CMyMaxPath::SetSearchParameters(int sc_num, int sc_start, int sc_step, int sp_step, int radius, int ra_num, float ra_start, float ra_step)
{
	nNumOfScales = sc_num;
	nStartScale = sc_start;
	nScaleStep = sc_step;
	nSpacialStep = sp_step;
	nRadius = radius;
	nNumOfRatios = ra_num;
	fSmallestRatio = ra_start;
	fRatioStep = ra_step;
}
CMyMaxPath::~CMyMaxPath(void)
{
	if (pPointList!=NULL)
		free(pPointList);
}

void CMyMaxPath::InitializeData(int n, int w, int h, int l, IPOINT* pList)
{
	int i;
	nNumOfPoints = n;
	width = w;
	height = h;
	length = l;
	pPointList = (IPOINT*)malloc(sizeof(IPOINT)*n);
	for (i = 0; i < n; i++)
		pPointList[i] = pList[i];
}

void CMyMaxPath::Build3DTrellis(float small_neg)
{
	int t, i, j, x, y, w2, h2, xc, yc, left, top, right, bottom;
	CMy2DArray<float> arr;
	CMyIntegralImage<float> imgIntegral;	
	IPOINT pt;
	float localScore;

	trellis_width = int(width/nSpacialStep)+1;
	trellis_height = int(height/nSpacialStep)+1;
	for (i = 0; i < nNumOfScales; i++)
		for (j = 0; j < nNumOfRatios; j++)
			arrTrellis[i][j].SetSize(trellis_width, trellis_height, length);

	arr.SetSize(width, height);
	for (t=0; t < length; t++)
	{
		arr.SetAllValues(small_neg);
		for (i = 0; i < nNumOfPoints; i++)
		{
			pt = pPointList[i];
			if (pt.t == t)
			{
				arr.SetValue(pt.x, pt.y, pt.score);
			}
		}
		imgIntegral.ClearIntegral();
		imgIntegral.PrecomputeIntegralImage(arr);
		for (i = 0; i < nNumOfScales; i++)
		{
			h2 = (nStartScale + i*nScaleStep);
			for (j = 0; j < nNumOfRatios; j++)
			{
				w2 = int((fSmallestRatio+j*fRatioStep)*h2);
				for (y = 0; y < trellis_height; y++)
					for (x = 0; x < trellis_width; x++)
					{
						xc = nSpacialStep*x;
						yc = nSpacialStep*y;
						left = xc-w2+1;
						top = yc-h2+1;
						right = xc+w2;
						bottom = yc+h2;
						if (left>=0&&top>=0&&right<width&&bottom<height)
							localScore = imgIntegral.IntegralSum(left,top,right,bottom);
						else
							localScore = GLOBAL_NEG;
						arrTrellis[i][j].SetValue(x,y,t,localScore);
					}
			}
		}
	}
}

float CMyMaxPath::MaxPathSearch(CMySparseVolume *vol, MY2DBOX* pOutputBoxes, float small_neg)
{
	InitializeData(vol->GetNumOfPoints(), vol->GetWidth(), vol->GetHeight(), vol->GetLength(), vol->GetPointListRef());
	Build3DTrellis(small_neg);
	MessageForwarding();
	PathTraceBack(pOutputBoxes);
	return sum_best;
}
float CMyMaxPath::MaxPathSearch(CMySparseVolume *vol, CMyPath &path, float small_neg)
{
	MY2DBOX *arrBox;
	InitializeData(vol->GetNumOfPoints(), vol->GetWidth(), vol->GetHeight(), vol->GetLength(), vol->GetPointListRef());
	Build3DTrellis(small_neg);
	MessageForwarding();
	arrBox = (MY2DBOX*)malloc(sizeof(MY2DBOX)*vol->GetLength());
	PathTraceBack(arrBox);
	path.Initialize(vol->GetLength(), arrBox);
	free(arrBox);
	return sum_best;
}

void CMyMaxPath::ClearAll()
{
	int i, j;
	for (i = 0; i < nNumOfScales; i++)
		for (j = 0; j < nNumOfRatios; j++)
		{
			arrTrellis[i][j].ClearAll();
			arrPrevX[i][j].ClearAll();
			arrPrevY[i][j].ClearAll();
			arrPrevS[i][j].ClearAll();
			arrPrevR[i][j].ClearAll();
		}
	nNumOfScales = 0;
	nNumOfRatios = 0;
}
void CMyMaxPath::MessageForwarding(void)
{
	int i, j, x, y, t, x0, y0, s0, r0;
	float sum_max, tmp;
	int x_max, y_max, scale_max, ratio_max, x1, x2, y1, y2, s1, s2, r1, r2;
	CMy2DArray<float> arrCurSum[MAX_SCALE][MAX_RATIO],arrPreSum[MAX_SCALE][MAX_RATIO];
	for (i = 0; i < nNumOfScales; i++)
		for (j = 0; j < nNumOfRatios; j++)
		{
			arrPrevX[i][j].SetSize(trellis_width, trellis_height, length);
			arrPrevY[i][j].SetSize(trellis_width, trellis_height, length);
			arrPrevS[i][j].SetSize(trellis_width, trellis_height, length);
			arrPrevR[i][j].SetSize(trellis_width, trellis_height, length);
			arrCurSum[i][j].SetSize(trellis_width, trellis_height);
			arrPreSum[i][j].SetSize(trellis_width, trellis_height);
		}
	for (i = 0; i < nNumOfScales; i++)
		for (j = 0; j < nNumOfRatios; j++)
		{
			arrPrevX[i][j].SetAllValues(-1);
			arrPrevY[i][j].SetAllValues(-1);
			arrPrevS[i][j].SetAllValues(-1);
			arrPrevR[i][j].SetAllValues(-1);
		}
	sum_best = -10^5;
	x_best = y_best = t_best = scale_best = ratio_best = -1;
	for (i = 0; i < nNumOfScales; i++)
		for (j = 0; j < nNumOfRatios; j++)
			for (y = 0; y < trellis_height; y++)
				for (x = 0; x < trellis_width; x++)
					arrPreSum[i][j].SetValue(x,y,arrTrellis[i][j].GetValue(x,y,0));

	for (t = 1; t < length; t++)
	{
		for (i = 0; i < nNumOfScales; i++)
			for (j = 0; j < nNumOfRatios; j++)
				for (y = 0; y < trellis_height; y++)
					for (x = 0; x < trellis_width; x++)
					{
						sum_max = -10^5;
						r1 = max(0,j-1);
						r2 = min(nNumOfRatios-1,j+1);
						s1 = max(0,i-1);
						s2 = min(nNumOfScales-1,i+1);
						y1 = max(0,y-nRadius);
						y2 = min(trellis_height-1,y+nRadius);
						x1 = max(0,x-nRadius);
						x2 = min(trellis_width-1,x+nRadius);
						for (s0=s1; s0<=s2; s0++)
							for (r0=r1; r0<=r2; r0++)
								for (y0=y1; y0<=y2; y0++)
									for (x0=x1; x0<=x2; x0++)
									{
										tmp = arrPreSum[s0][r0].GetValue(x0,y0);
										if (tmp>sum_max)
										{
											sum_max = tmp;
											x_max = x0;
											y_max = y0;
											scale_max = s0;
											ratio_max = r0;
										}
									}
						if (sum_max>0)
						{
							arrCurSum[i][j].SetValue(x,y,sum_max+arrTrellis[i][j].GetValue(x,y,t));
							arrPrevX[i][j].SetValue(x,y,t,x_max);
							arrPrevY[i][j].SetValue(x,y,t,y_max);
							arrPrevS[i][j].SetValue(x,y,t,scale_max);
							arrPrevR[i][j].SetValue(x,y,t,ratio_max);
						}
						else
							arrCurSum[i][j].SetValue(x,y,arrTrellis[i][j].GetValue(x,y,t));
						tmp = arrCurSum[i][j].GetValue(x,y);
						if (tmp > sum_best)
						{
							sum_best = tmp;
							x_best = x;
							y_best = y;
							t_best = t;
							scale_best = i;
							ratio_best = j;
						}
					}
		for (i = 0; i < nNumOfScales; i++)
			for (j = 0; j < nNumOfRatios; j++)
				arrPreSum[i][j] = arrCurSum[i][j];
	}
}

void CMyMaxPath::PathTraceBack(MY2DBOX* pOutBoxes)
{
	int i, x, y, t, s, r, w2, h2, x1, y1, s1, r1;
	MY2DBOX empty = {0,0,0,0};
	for (i = 0; i < length; i++)
		pOutBoxes[i] = empty;
	x = x_best;
	y = y_best;
	t = t_best;
	s = scale_best;
	r = ratio_best;
	do{
		h2 = (nStartScale + s*nScaleStep);
		w2 = int((fSmallestRatio+r*fRatioStep)*h2);

		pOutBoxes[t].left = nSpacialStep*x-w2+1;
		pOutBoxes[t].top = nSpacialStep*y-h2+1;
		pOutBoxes[t].right = nSpacialStep*x+w2;
		pOutBoxes[t].bottom = nSpacialStep*y+h2;
		
		x1 = x;
		y1 = y;
		s1 = s;
		r1 = r;
		x = arrPrevX[s1][r1].GetValue(x1,y1,t);
		y = arrPrevY[s1][r1].GetValue(x1,y1,t);
		s = arrPrevS[s1][r1].GetValue(x1,y1,t);
		r = arrPrevR[s1][r1].GetValue(x1,y1,t);
		t = t-1;
	}while(x!=-1&&y!=-1&&s!=-1&&r!=-1);
}
bool CMyMaxPath::LoadSettingsFromFile(char* fn)
{
	FILE *f;
	fopen_s(&f, fn, "rt");
	if (f == NULL)
		return false;
	fscanf_s(f,"%d",&nNumOfScales);
	if (nNumOfScales<=0||nNumOfScales>MAX_SCALE)
		return false;
	fscanf_s(f,"%d",&nScaleStep);
	if (nScaleStep<=0)
		return false;
	fscanf_s(f,"%d",&nSpacialStep);
	if (nSpacialStep<=0)
		return false;
	fscanf_s(f,"%d",&nStartScale);
	if (nStartScale<=0)
		return false;
	fscanf_s(f,"%d",&nNumOfRatios);
	if (nNumOfRatios<=0||nNumOfRatios>MAX_RATIO)
		return false;
	fscanf_s(f,"%f",&fSmallestRatio);
	if (fSmallestRatio<=0)
		return false;
	fscanf_s(f,"%f",&fRatioStep);
	if (fRatioStep<=0)
		return false;
	fscanf_s(f,"%d",&nRadius);
	if (nRadius<=0)
		return false;
	fclose(f);
	return true;
}