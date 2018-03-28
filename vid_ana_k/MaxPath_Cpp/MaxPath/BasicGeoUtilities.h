#pragma once
#include "GlobalDefinitions.h"

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/

class CBasicGeoUtilities
{
public:
	CBasicGeoUtilities(void);
	~CBasicGeoUtilities(void);
	static bool PointInRect(int x, int y, int left, int top, int right, int bottom);
	bool IsEmptyBox(MY2DBOX box);
};
