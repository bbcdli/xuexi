#include "StdAfx.h"
#include "MySparseVolume.h"
#include "MyMaxPath.h"
#include <math.h>

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/


CMySparseVolume::CMySparseVolume(void)
{
	nNumOfPoints = 0;
	ptsList = NULL;
	pBoxList = NULL;
	label = -1;
}

CMySparseVolume::~CMySparseVolume(void)
{
	if (ptsList!=NULL)
		free(ptsList);
	if (pBoxList!=NULL)
		free(pBoxList);
}
void CMySparseVolume::RemoveAll()
{
	if (ptsList!=NULL)
		free(ptsList);
	if (pBoxList!=NULL)
		free(pBoxList);
	nNumOfPoints = 0;
	ptsList = NULL;
	pBoxList = NULL;
	label = -1;
}
bool CMySparseVolume::LoadFromFile(char* fn)
{
	FILE *f;
	int i, x, y, t;
	float tmp;

	fopen_s(&f, fn, "rt");
	if (f==NULL)
		return false;
	fscanf_s(f, "%d%d%d%d", &width, &height, &length, &nNumOfPoints);
	ptsList = (IPOINT*)malloc(nNumOfPoints*sizeof(IPOINT));
	if (ptsList==NULL)
		return false;
	for (i = 0; i < nNumOfPoints; i++)
	{
		fscanf_s(f, "%d%d%d%f", &x, &y, &t, &tmp);
		ptsList[i].x = x;
		ptsList[i].y = y;
		ptsList[i].t = t;
		ptsList[i].score = tmp;
	}
	fclose(f);
	return true;
}
bool CMySparseVolume::LoadAnnotation(char* fn)
{
	FILE *f;
	int left, top, right, bottom, i, x, y, t, tmp;
	CBasicGeoUtilities uti;

	fopen_s(&f, fn, "rt");
	if (f==NULL)
		return false;
	fscanf_s(f,"%d%d", &label, &tmp);
	if (tmp != length)
	{
		printf("the length does not match!");
		return false;
	}
	if (label==1)
	{
		pBoxList = (MY2DBOX*)malloc(sizeof(MY2DBOX)*length);
		if (pBoxList==NULL)
		{
			printf("Du, i have not enough memory!");
			return false;
		}
		for (i = 0; i < length; i++)
		{
			fscanf_s(f,"%d%d%d%d", &left, &top, &right, &bottom);
			pBoxList[i].left = left;
			pBoxList[i].top = top;
			pBoxList[i].right = right;
			pBoxList[i].bottom = bottom;
		}
		for (i = 0; i < nNumOfPoints; i++)
		{
			x = ptsList[i].x;
			y = ptsList[i].y;
			t = ptsList[i].t;
		}
	}
	return true;
}
IPOINT CMySparseVolume::GetPoint(int idx)
{
	if (idx<0||idx>=nNumOfPoints)
		printf("Du, index must be wrong!");
	return ptsList[idx];
}

int CMySparseVolume::GetNumOfPoints(void)
{
	return nNumOfPoints;
}

IPOINT* CMySparseVolume::GetPointListRef(void)
{
	return ptsList;
}

MY2DBOX* CMySparseVolume::GetBoxList(void)
{
	return pBoxList;
}