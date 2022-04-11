/**
 * @file LineMandelCalculator.cc
 * @author Vít Barták <xbarta47@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 07.11.2021
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <immintrin.h>

#include <stdlib.h>

#include "LineMandelCalculator.h"

LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize, unsigned limit) : BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	dataX = (float *)(_mm_malloc(height * width * sizeof(float), 64));
	dataY = (float *)(_mm_malloc(height * width * sizeof(float), 64));

	for (int i = 0; i < height; i++) // Inicializace polí
	{
		int *const pdata = data + i * width;
		float *const arrdataX = dataX + i * width;
		float *const arrdataY = dataY + i * width;

#pragma omp simd aligned(pdata : 64, arrdataX : 64, arrdataY : 64)
		for (int j = 0; j < width; j++)
		{
			pdata[j] = limit;
			arrdataX[j] = x_start + j * dx;
			arrdataY[j] = y_start + i * dy;
		}
	}
}

LineMandelCalculator::~LineMandelCalculator()
{
	_mm_free(data);
	_mm_free(dataX);
	_mm_free(dataY);
	data = NULL;
	dataX = NULL;
	dataY = NULL;
}

int *LineMandelCalculator::calculateMandelbrot()
{
	int it = 0; // Iterátor pro kontrolu pokračování výpočtu
	for (int i = 0; i < height; i++)
	{
		float imag = y_start + i * dy;
		it = 0;

		int *const pdata = data + i * width;
		float *const arrdataX = dataX + i * width;
		float *const arrdataY = dataY + i * width;

		for (int k = 0; it < width && k < limit; ++k)
		{
#pragma omp simd aligned(pdata : 64, arrdataX : 64, arrdataY : 64) reduction(+ : it)
			for (int j = 0; j < width; j++)
			{
				float real = x_start + j * dx;

				float zReal = arrdataX[j];
				float zImag = arrdataY[j];

				float r2 = zReal * zReal;
				float i2 = zImag * zImag;

				if (r2 + i2 > 4.0f && pdata[j] == limit)
				{
					pdata[j] = k;
					it += 1;
				}

				arrdataY[j] = 2.0f * zReal * zImag + imag;
				arrdataX[j] = r2 - i2 + real;
			}
		}
	}
	return data;
}