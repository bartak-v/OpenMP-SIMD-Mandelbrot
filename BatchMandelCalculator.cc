/**
 * @file BatchMandelCalculator.cc
 * @author Vít Barták <xbarta47@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 12.11.2021
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <immintrin.h>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit) : BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
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

BatchMandelCalculator::~BatchMandelCalculator()
{
	_mm_free(data);
	_mm_free(dataX);
	_mm_free(dataY);
	data = NULL;
	dataX = NULL;
	dataY = NULL;
}

int *BatchMandelCalculator::calculateMandelbrot()
{

	const int blockSize = 64;
	for (int blockN = 0; blockN < height / blockSize; blockN++)
	{
		for (int blockP = 0; blockP < width / blockSize; blockP++)
		{
			int it = 0; // Iterátor pro kontrolu pokračování výpočtu
			for (int i = 0; i < blockSize; i++)
			{
				int iGlobal = blockN * blockSize + i;

				float imag = y_start + iGlobal * dy;
				it = 0;

				int *const pdata = data + iGlobal * width;
				float *const arrdataX = dataX + iGlobal * width;
				float *const arrdataY = dataY + iGlobal * width;

				for (int k = 0; it < blockSize && k < limit; ++k)
				{
#pragma omp simd aligned(pdata : 64, arrdataX : 64, arrdataY : 64) reduction(+ \
																			 : it)
					for (int j = 0; j < blockSize; j++)
					{
						int jGlobal = blockP * blockSize + j;
						float real = x_start + jGlobal * dx;

						float zReal = arrdataX[jGlobal];
						float zImag = arrdataY[jGlobal];

						float r2 = zReal * zReal;
						float i2 = zImag * zImag;

						if (r2 + i2 > 4.0f && pdata[jGlobal] == limit)
						{
							pdata[jGlobal] = k;
							it += 1;
						}

						arrdataY[jGlobal] = 2.0f * zReal * zImag + imag;
						arrdataX[jGlobal] = r2 - i2 + real;
					}
				}
			}
		}
	}
	return data;
}