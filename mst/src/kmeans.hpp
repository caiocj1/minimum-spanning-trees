#include <iostream>
#include <cassert>
#include <cmath>   // for sqrt, fabs
#include <cfloat>  // for DBL_MAX
#include <cstdlib> // for rand, srand
#include <ctime>   // for rand seed
#include <fstream>
#include <cstdio> // for EOF
#include <string>
#include <algorithm> // for count
#include <vector>
#include <unordered_map>

using std::rand;
using std::srand;
using std::time;

class point
{
public:
	static int d;
	double* coords;
	int label;

	point()
	{
		label = 0;
		coords = new double[d];
		for (int i = 0; i < d; i++)
			coords[i] = 0;
	}

	~point() { delete[] coords; }

	void print() const
	{
		for (int i = 0; i < d; i++)
		{
			std::cout << coords[i];
			if (i < d - 1)
				std::cout << "\t";
		}
		std::cout << "\n";
	}

	double squared_dist(const point& q) const
	{
		double res = 0;
		for (int i = 0; i < d; i++)
			res += (coords[i] - q.coords[i]) * (coords[i] - q.coords[i]);
		return res;
	}
};

int point::d;

class cloud
{
private:
	int d;
	int n;
	int k;

	point* points;
	point* centers;

public:
	cloud(std::vector<std::vector<double>> data, int _d, int _n, int _k)
	{
		d = _d;
		point::d = _d;
		k = _k;

		n = _n;

		points = new point[n];
		centers = new point[k];

		for (int i = 0; i < n; i++)
			for (int j = 0; j < d; j++)
				points[i].coords[j] = data[i][j];

		srand(time(0));
	}

	~cloud()
	{
		delete[] centers;
		delete[] points;
	}

	int get_d() const
	{
		return d;
	}

	int get_n() const
	{
		return n;
	}

	int get_k() const
	{
		return k;
	}

	point& get_point(int i)
	{
		return points[i];
	}

	point& get_center(int j)
	{
		return centers[j];
	}

	void set_center(const point& p, int j)
	{
		for (int m = 0; m < d; m++)
			centers[j].coords[m] = p.coords[m];
	}

	double intracluster_variance() const
	{
		double res = 0;

		for (int i = 0; i < n; i++)
		{
			res += points[i].squared_dist(centers[points[i].label]);
		}
		return res / n;
	}

	int set_voronoi_labels()
	{
		int res = 0;

		for (int i = 0; i < n; i++)
			for (int j = 0; j < k; j++)
			{
				double dist = points[i].squared_dist(centers[j]);
				double cur = points[i].squared_dist(centers[points[i].label]);
				if (dist <= cur)
				{
					int min = std::min(j, points[i].label);
					if (points[i].label != min)
					{
						points[i].label = min;
						res++;
					}
				}
			}

		return res;
	}

	void set_centroid_centers()
	{
		double** centroids = new double* [k];
		for (int i = 0; i < k; i++)
		{
			centroids[i] = new double[d];
			for (int j = 0; j < d; j++)
				centroids[i][j] = 0;
		}

		int* counter = new int[k];
		for (int i = 0; i < k; i++)
			counter[i] = 0;

		for (int i = 0; i < n; i++)
		{
			counter[points[i].label]++;
			for (int j = 0; j < d; j++)
				centroids[points[i].label][j] += points[i].coords[j];
		}

		for (int i = 0; i < k; i++)
			for (int j = 0; j < d; j++)
				if (counter[i] > 0)
					centroids[i][j] /= (float)counter[i];

		for (int i = 0; i < k; i++)
			if (counter[i] > 0)
				for (int j = 0; j < d; j++)
					centers[i].coords[j] = centroids[i][j];
	}

	void init_closest_center()
	{
		for (int i = 0; i < n; i++)
		{
			int min = 0;
			for (int j = 0; j < k; j++)
			{
				if (i != j && points[i].squared_dist(centers[j]) < points[i].squared_dist(centers[min]))
					min = j;
			}
			points[i].label = min;
		}
	}

	void lloyd()
	{
		init_forgy();
		init_closest_center();

		int changed;
		do
		{
			set_centroid_centers();
			changed = set_voronoi_labels();
		} while (changed > 0);
	}

	void init_forgy()
	{
		bool* checked = new bool[n];
		for (int i = 0; i < n; i++)
			checked[i] = false;

		for (int i = 0; i < k; i++)
		{
			int new_center = rand() % n;
			while (checked[new_center])
				new_center = rand() % n;
			checked[new_center] = true;

			for (int j = 0; j < d; j++)
				centers[i].coords[j] = points[new_center].coords[j];
		}
	}
};
