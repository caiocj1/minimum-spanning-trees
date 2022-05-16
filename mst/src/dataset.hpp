#pragma once

#include <vector>
#include <string>

class Dataset {
	int n, d;
	std::vector<std::vector<double>> data;
public:
	Dataset(int n, int d);
	Dataset(std::string filename);

	int get_n();
	int get_d();
	std::vector<double> get_instance(int i);

	static double dist(std::vector<double> d1, std::vector<double> d2);

	std::vector<int> k_means(int k);

	void print();

	void rescale();
	void standardize();
};