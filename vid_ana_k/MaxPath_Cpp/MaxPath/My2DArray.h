#pragma once

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/

template <class T>
class CMy2DArray
{
private:
	T *buff;
	int width, height;
public:
	int GetWidth() { return width;}
	int GetHeight() { return height;}
	void SetSize(int w, int h)
	{
		width = w; height = h;
		if (buff)
			free(buff);
		buff = (T*)malloc(sizeof(T)*w*h);
		if (buff==NULL)
			printf("Du, I don't have enough memory!");
		memset(buff, 0, sizeof(T)*w*h);
	}
	T GetValue(int x, int y)
	{
		if (x<0||x>=width||y<0||y>=height)
			printf("(x=%d, y=%d w=%d, h=%d) Du, get wrong 2D array index!", x, y, width, height);
		return buff[y*width+x];
	}
	void SetValue(int x, int y, T value)
	{
		if (x<0||x>=width||y<0||y>=height)
			printf("(x=%d, y=%d w=%d, h=%d) Du, set wrong 2D array index!", x, y, width, height);
		buff[y*width+x] = value;
	}
	void ClearAll()
	{
		if (buff)
			free(buff);
		buff = NULL;
		width = height = 0;
	}
	CMy2DArray<T> operator=(CMy2DArray<T> &arr)
	{
		if (buff)
			free(buff);
		width = arr.GetWidth(); 
		height = arr.GetHeight(); 
		buff = (T*)malloc(sizeof(T)*width*height);
		memcpy(buff, arr.buff, sizeof(T)*width*height);
		return *this;
	}
	CMy2DArray(CMy2DArray<T> &arr)
	{
		width = arr.GetWidth(); 
		height = arr.GetHeight(); 
		buff = (T*)malloc(sizeof(T)*width*height);
		memcpy(buff, arr.buff, sizeof(T)*width*height);
	}
	CMy2DArray(void)
	{
		buff = NULL;
		width = height = 0;
	}
	~CMy2DArray(void)
	{
		if (buff)
			free(buff);
	}
	void SetAllValues(T v)
	{
		int i, j;
		for (i = 0 ; i < height; i++)
			for (j = 0; j < width; j++)
				SetValue(j,i,v);
	}
};
