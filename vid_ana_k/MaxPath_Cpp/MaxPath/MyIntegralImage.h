#pragma once
#include "My2DArray.h"

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/


template<class T>
class CMyIntegralImage
{
private:
	CMy2DArray<T> integralImg;
public:
	void PrecomputeIntegralImage(CMy2DArray<T> imgInput)
	{
		int w, h;
		int x, y;
		w = imgInput.GetWidth();
		h = imgInput.GetHeight();
		integralImg.SetSize(w,h);
		integralImg.SetValue(0,0,imgInput.GetValue(0,0));
		for (y = 1; y < h; y++)
			integralImg.SetValue(0,y,integralImg.GetValue(0,y-1)+imgInput.GetValue(0,y));
		for (x = 1; x < w; x++)
			integralImg.SetValue(x,0,integralImg.GetValue(x-1,0)+imgInput.GetValue(x,0));
		for (y = 1 ; y < h; y++)
			for (x = 1; x < w; x++)
			integralImg.SetValue(x,y,integralImg.GetValue(x-1,y)+integralImg.GetValue(x,y-1)-integralImg.GetValue(x-1,y-1)+
										imgInput.GetValue(x,y));
	}
	T IntegralSum(int x1, int y1, int x2, int y2)
	{
		int w, h;
		T s;
		w = integralImg.GetWidth();
		h = integralImg.GetHeight();
		if (x1 < 0 || x1 >= w || y1 < 0 || y1 >= h)
			printf("Du, wrong 2D array index on integral image");
		if (x2 < 0 || x2 >= w || y2 < 0 || y2 >= h)
			printf("Du, wrong 2D array index on integral image");
		if (x1 > x2 || y1 > y2)
			printf("Du, wrong subwindow index for computing integral image sum");
		s = integralImg.GetValue(x2,y2);
		if (x1>0)
			s -= integralImg.GetValue(x1-1,y2);
		if (y1>0)
			s -= integralImg.GetValue(x2,y1-1);
		if (x1>0&&y1>0)
			s += integralImg.GetValue(x1-1,y1-1);
		return s;
	}
	void ClearIntegral()
	{
		integralImg.ClearAll();
	}
	CMyIntegralImage(void)
	{};
	~CMyIntegralImage(void)
	{};
};
