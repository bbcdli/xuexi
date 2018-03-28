#pragma once
#include "GlobalDefinitions.h"
#include <stdlib.h>

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/

class CMyPath
{
private:
	int nLength;
	MY2DBOX *arrBox;
	float path_score;
public:
	void SetPathScore(float s) {path_score = s;};
	float GetPathScore() {return path_score;};
	int GetLength() {return nLength;};
	MY2DBOX* GetBoxPointer(){return arrBox;};
	CMyPath(void);
	CMyPath(CMyPath &other);
	~CMyPath(void);
	void Initialize(int length, MY2DBOX* arr);
	void RemoveAll(void);
	int operator==(CMyPath &other);
	CMyPath operator=(CMyPath &other);
	int GetNumberOfNonEmptyBox(void);
};
