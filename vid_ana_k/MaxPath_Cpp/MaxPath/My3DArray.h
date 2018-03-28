#pragma once
#include <memory.h>

/*
Author: Du Tran <dutran@ieee.org> more at: www.dutran.org
Date: 06 June 2011 
More details: please be refered to read
Du Tran and Junsong Yuan, Optimal Spatio-Temporal Path Discovery for
Video Event Detection, CVPR 2011.
*/


template<class T>
class CMy3DArray
{
private:
	T* buff;
	int width, height, depth;
public:
	int GetWidth() { return width;}
	int GetHeight() { return height;}
	int GetDepth() { return depth;}
	void SetSize(int w, int h, int d)
	{
		width = w; height = h; depth = d;
		if (buff)
			free(buff);
		buff = (T*)malloc(sizeof(T)*w*h*d);
		if (buff==NULL)
			printf("Du, I don't have enough memory!");
		memset(buff, 0, sizeof(T)*w*h*d);
	}
	CMy3DArray<T> GetSubVolume(int x1, int x2, int y1, int y2, int z1, int z2)
	{
		int tmp, i, j, k;
		if (x1>x2) {tmp = x1; x1 = x2; x2 = tmp;}
		if (y1>y2) {tmp = y1; y1 = y2; y2 = tmp;}
		if (z1>z2) {tmp = z1; z1 = z2; z2 = tmp;}
		CMy3DArray<T> subVolume;
		subVolume.SetSize(x2-x1+1,y2-y1+1,z2-z1+1);
		for (i = z1; i <= z2; i++)
			for (j = y1; j <= y2; j++)
				for (k = x1; k <= x2; k++)
					subVolume.SetValue(k-x1, j-y1, i-z1, GetValue(k, j, i));
		return subVolume;
	}

	bool SaveVolume(char* fn)
	{
		FILE *f;
		fopen_s(&f, fn, "wb");
		if (!f)
			return FALSE;
		fwrite(&width, sizeof(int), 1, f);
		fwrite(&height, sizeof(int), 1, f);
		fwrite(&depth, sizeof(int), 1, f);
		
		fwrite(buff, sizeof(T), width*height*depth, f);
		fclose(f);
		return TRUE;
	}
	T GetValue(int x, int y, int z)
	{
		if (x<0||x>=width||y<0||y>=height||z<0||z>=depth)
			printf("Du, get a wrong 3D array index!");
		return buff[z*width*height+y*width+x];
	}
	void SetValue(int x, int y, int z, T value)
	{
		if (x<0||x>=width||y<0||y>=height||z<0||z>=depth)
			printf("Du, set a wrong 3D array index!");
		buff[z*width*height+y*width+x] = value;
	}
	void ClearAll()
	{
		if (buff)
			free(buff);
		buff = NULL;
		width = height = depth = 0;
	}
	CMy3DArray<T> operator=(CMy3DArray<T> &volume)
	{
		if (buff)
			free(buff);
		width = volume.GetWidth(); 
		height = volume.GetHeight(); 
		depth = volume.GetDepth();
		buff = (T*)malloc(sizeof(T)*width*height*depth);
		memcpy(buff, volume.buff, sizeof(T)*width*height*depth);
		return *this;
	}
	CMy3DArray(CMy3DArray<T> &volume)
	{
		width = volume.GetWidth(); 
		height = volume.GetHeight(); 
		depth = volume.GetDepth();
		buff = (T*)malloc(sizeof(T)*width*height*depth);
		memcpy(buff, volume.buff, sizeof(T)*width*height*depth);
	}
	CMy3DArray(void)
	{
		buff = NULL;
		width = height = depth = 0;
	}
	~CMy3DArray(void)
	{
		if (buff)
			free(buff);
	}
	void SetAllValues(T v)
	{
		int x, y, t;
		for (t = 0; t < depth; t++)
			for (y = 0 ; y < height; y++)
				for (x = 0; x < width; x++)
					SetValue(x,y,t,v);
	}
	void Set1SliceSameValue(T v, int t)
	{
		int x, y;
		for (y = 0 ; y < height; y++)
			for (x = 0; x < width; x++)
				SetValue(x,y,t,v);
	}
};
