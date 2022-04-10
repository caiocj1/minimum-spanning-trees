#pragma once
#include <list>

class Graph
{
	int n;
	std::list<std::pair<int, float>>* adj;
public:
	Graph(int n_);
	~Graph();

	int get_n();

	void print();

	void add_bi_edge(int u, int v, float w);
};