#include "StdAfx.h"
#include "BasicGeoUtilities.h"
#include <math.h>

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/


CBasicGeoUtilities::CBasicGeoUtilities(void)
{
}

CBasicGeoUtilities::~CBasicGeoUtilities(void)
{
}

bool CBasicGeoUtilities::PointInRect(int x, int y, int left, int top, int right, int bottom)
{
	if (x<left)
		return false;
	if (x>right)
		return false;
	if (y<top)
		return false;
	if (y>bottom)
		return false;
	return true;
}

bool CBasicGeoUtilities::IsEmptyBox(MY2DBOX box)
{
	if (box.right == box.left || box.top == box.bottom)
		return true;
	return false;
}