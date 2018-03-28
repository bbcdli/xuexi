#pragma once
#include<stdlib.h>
#include<stdio.h>
#include "BasicGeoUtilities.h"
#include "GlobalDefinitions.h"
#include "MyPath.h"

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/

class CMySparseVolume
{
private:
	int width, height, length;
	int nNumOfPoints;
	IPOINT *ptsList;
	short label;
	MY2DBOX* pBoxList;
public:
	int GetWidth() {return width;}
	int GetHeight() {return height;}
	int GetLength() {return length;}
	short GetLabel() {return label;}
	CMySparseVolume(void);
	~CMySparseVolume(void);
	bool LoadFromFile(char* fn);
	bool LoadAnnotation(char* fn);
	IPOINT GetPoint(int idx);
	int GetNumOfPoints(void);
	IPOINT* GetPointListRef(void);
	MY2DBOX* GetBoxList(void);
	void RemoveAll();
};
