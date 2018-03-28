#pragma once
#define MAX_EXAMPLES 200
/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/

typedef struct tagIPOINT
{
	int x,y,t;
	float score;
}IPOINT;

typedef struct tagMY2DBOX
{
	int left, top, right, bottom;
}MY2DBOX;
