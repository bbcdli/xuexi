#pragma once
#include "GlobalDefinitions.h"
#include "My3DArray.h"
#include "My2DArray.h"
#include "MySparseVolume.h"

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/

#define MAX_SCALE 12
#define MAX_RATIO 10
#define SMALL_NEG 0
#define GLOBAL_NEG -10


#define NUM_OF_SCALES 10
#define SCALE_STEP 4
#define SPACIAL_STEP 2
#define START_SCALE 18
#define NUM_OF_RATIOS 5
#define SMALLEST_RATIO float(0.3)
#define RATIO_STEP float(0.1)
#define NRADIUS 3



class CMyMaxPath
{
private:
	int width, height, length;
	int nNumOfPoints;
	IPOINT *pPointList;
	int nNumOfScales, nScaleStep, nSpacialStep, nStartScale;
	int nNumOfRatios, nRadius;
	float fSmallestRatio, fRatioStep;
	CMy3DArray<float> arrTrellis[MAX_SCALE][MAX_RATIO];
	CMy3DArray<short> arrPrevX[MAX_SCALE][MAX_RATIO], arrPrevY[MAX_SCALE][MAX_RATIO], arrPrevS[MAX_SCALE][MAX_RATIO], arrPrevR[MAX_SCALE][MAX_RATIO];
	int trellis_width, trellis_height;
	float sum_best;
	int x_best, y_best, t_best, scale_best, ratio_best;
public:
	CMyMaxPath(void);
	~CMyMaxPath(void);
	void InitializeData(int n, int w, int h, int l, IPOINT* pList);
	void Build3DTrellis(float small_neg = SMALL_NEG);
	float MaxPathSearch(CMySparseVolume *vol, MY2DBOX* pOutputBoxes, float small_neg);
	float MaxPathSearch(CMySparseVolume *vol, CMyPath &path, float small_neg);
	void ClearAll();
	void MessageForwarding(void);
	void PathTraceBack(MY2DBOX* pOutBoxes);
	void SetSearchParameters(int sc_num = NUM_OF_SCALES, int sc_start = START_SCALE, int sc_step = SCALE_STEP, 
						     int sp_step = SPACIAL_STEP, int radius = NRADIUS, int ra_num = NUM_OF_RATIOS, 
							 float ra_start = SMALLEST_RATIO, float ra_step = RATIO_STEP);
	bool LoadSettingsFromFile(char* fn);
};
