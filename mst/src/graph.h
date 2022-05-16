#pragma once
#include <set>
#include <tuple>
#include <vector>
#include "dataset.hpp"
#include <mpi.h>

typedef std::pair<int, double> edge;
typedef std::tuple<double, int, int> bi_edge;

class Graph
{
	int n, m;
	std::set<edge>* adj;
public:
	Graph(int n_);
	Graph(Dataset data, std::string normalize = "none");
	~Graph();

	int get_n();
	std::set<edge>* get_adj();

	void print();

	void add_bi_edge(int u, int v, double w);

	static Graph* random_complete_graph(int n_);

	Graph* prim();

	Graph* master_parallel_prim();
	static void slave_parallel_prim();

	Graph* boruvska();
	Graph* kruskal();

	std::vector<int> mst_cluster(int k);
	std::vector<int> mst_cluster();
};