#pragma once
#include <list>

typedef std::pair<int, float> edge;

class Graph
{
	int n;
	std::list<edge>* adj;
public:
	Graph(int n_);
	~Graph();

	int get_n();

	void print();

	void add_bi_edge(int u, int v, float w);

	Graph* prim();
};