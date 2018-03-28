#include "StdAfx.h"
#include "MyPath.h"

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/


CMyPath::CMyPath(void)
{
	path_score = 0;
	arrBox = NULL;
}

CMyPath::~CMyPath(void)
{
	if (arrBox!=NULL)
		free(arrBox);
}
CMyPath::CMyPath(CMyPath &other)
{
	int i;
	nLength = other.nLength;
	if (nLength)
	{
		arrBox = (MY2DBOX*)malloc(sizeof(MY2DBOX)*nLength);
		if (arrBox == NULL)
		{
			printf("Du, not enough memory!!!");
			return;
		}
		for (i = 0; i < nLength; i++)
			arrBox[i] = other.arrBox[i];
	}
	path_score = other.path_score;
}
CMyPath CMyPath::operator=(CMyPath& other)
{
	int i;
	nLength = other.nLength;
	if (nLength)
	{
		arrBox = (MY2DBOX*)malloc(sizeof(MY2DBOX)*nLength);
		if (arrBox == NULL)
		{
			printf("Du, not enough memory!!!");
		}
		for (i = 0; i < nLength; i++)
			arrBox[i] = other.arrBox[i];
	}
	path_score = other.path_score;
	return (*this);
}
void CMyPath::Initialize(int length, MY2DBOX* arr)
{
	nLength = length;
	arrBox = (MY2DBOX*)malloc(sizeof(MY2DBOX)*nLength);
	if (arrBox==NULL)
	{
		printf("Du, not enough memory!!!");
		return;
	}
	for (int i = 0; i < length; i++)
		arrBox[i] = arr[i];
}

void CMyPath::RemoveAll(void)
{
	if (arrBox!=NULL)
		free(arrBox);
	arrBox = NULL;
	nLength = 0;
}
int CMyPath::operator==(CMyPath &other)
{
	int i;
	if (other.nLength != nLength)
		return 0;
	for (i = 0; i < nLength; i++)
	{
		if (arrBox[i].left != other.arrBox[i].left)
			return 0;
		if (arrBox[i].right != other.arrBox[i].right)
			return 0;
		if (arrBox[i].top != other.arrBox[i].top)
			return 0;
		if (arrBox[i].bottom != other.arrBox[i].bottom)
			return 0;
	}
	return 1;
}
int CMyPath::GetNumberOfNonEmptyBox(void)
{
	int nonempty = 0;
	int i;
	for (i = 0; i < nLength; i++)
	{
		if (arrBox[i].left||arrBox[i].top||arrBox[i].right||arrBox[i].bottom)
			nonempty++;
	}	
	return nonempty;
}
