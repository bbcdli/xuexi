#pragma once
#include "MyMaxPath.h"
#include "MySparseVolume.h"
#include "GlobalDefinitions.h"

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/

class CMyMaxPathTester
{
public:
	CMyMaxPathTester(void);
	~CMyMaxPathTester(void);
	bool TestOneVolume(char* fn_sparse_vol, char* fn_settings, char* fn_output, float small_neg);
};
