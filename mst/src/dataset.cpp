#include "dataset.hpp"
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

/**
* Returns dataset with specified size and dimensions,
* with zeroes in every entry.
* 
* @param: number n of points, number d of dimensions
*/
Dataset::Dataset(int n_, int d_) : n(n_), d(d_), data(n_, std::vector<double>(d, 0)) { };

Dataset::Dataset(std::string filename)
{
	std::ifstream file;
	file.open(filename);
	assert(file.is_open());
	
	std::vector<double> row;
	std::string line, entry;

	std::getline(file, line);

	while (std::getline(file, line))
	{
		row.clear();

		std::stringstream str(line);
		while (std::getline(str, entry, ','))
			row.push_back(std::stod(entry));

		data.push_back(row);
	}

	n = data.size();
	d = data[0].size();
}

int Dataset::get_n()
{
	return n;
}

int Dataset::get_d()
{
	return d;
}

void Dataset::print()
{
	for (auto v : data)
	{
		for (int j = 0; j < d; j++)
			std::cout << v[j] << ", ";
		std::cout << std::endl;
	}
}

std::vector<double> Dataset::get_instance(int i)
{
	assert(i >= 0 and i < n);
	return data[i];
}

double Dataset::dist(std::vector<double> d1, std::vector<double> d2)
{
	assert(d1.size() == d2.size());

	double sum = 0;
	for (int i = 0; i < d1.size(); i++)
		sum += (d1[i] - d2[i]) * (d1[i] - d2[i]);
	return std::sqrt(sum);
}

void Dataset::rescale()
{
	std::vector<double> max(d, 0);
	std::vector<double> min(d, 0);

	for (int j = 0; j < d; j++)
	{
		max[j] = data[0][j];
		min[j] = max[j];
		
		for (int i = 0; i < n; i++)
		{
			if (data[i][j] > max[j])
				max[j] = data[i][j];
			if (data[i][j] < min[j])
				min[j] = data[i][j];
		}
	}

	for (int j = 0; j < d; j++)
		for (int i = 0; i < n; i++)
			data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
}

void Dataset::standardize()
{
	std::vector<double> avg(d, 0);
	std::vector<double> std_err(d, 0);

	for (int j = 0; j < d; j++)
	{
		double sum = 0;

		for (int i = 0; i < n; i++)
			sum += data[i][j];

		avg[j] = sum / n;
	}

	for (int j = 0; j < d; j++)
	{
		double sum = 0;

		for (int i = 0; i < n; i++)
			sum += (data[i][j] - avg[j]) * (data[i][j] - avg[j]);

		std_err[j] = std::sqrt(sum / n);
	}

	for (int j = 0; j < d; j++)
		for (int i = 0; i < n; i++)
			data[i][j] = (data[i][j] - avg[j]) / std_err[j];
}