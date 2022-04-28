#pragma once
#include <list>
#include <tuple>

typedef std::pair<int, double> edge;
typedef std::tuple<double, int, int> bi_edge;

class Graph
{
	int n, m;
	std::list<edge>* adj;
public:
	Graph(int n_);
	~Graph();

	int get_n();

	void print();

	void add_bi_edge(int u, int v, double w);

	Graph* prim();

	Graph* boruvska();

	Graph* kruskal();
};