#include "graph.h"
#include "union_find.hpp"
#include <stdexcept>
#include <iostream>
#include <queue>
#include <limits>
#include <random>
#include <assert.h>

Graph::Graph(int n_)
{
	n = n_;
	m = 0;
	adj = new std::set<edge>[n_];
}

Graph::Graph(Dataset data, std::string normalize)
{
	n = data.get_n();
	adj = new std::set<edge>[n];
	m = 0;

	if (normalize == "standardize")
		data.standardize();
	else if (normalize == "rescale")
		data.rescale();

	for (int i = 0; i < n; i++)
	{
		std::vector<double> x_i = data.get_instance(i);
		for (int j = i + 1; j < n; j++)
		{
			std::vector<double> x_j = data.get_instance(j);
			add_bi_edge(i, j, Dataset::dist(x_i, x_j));
		}
	}
}

Graph::~Graph()
{
	delete[] adj;
}

int Graph::get_n()
{
	return n;
}

void Graph::print()
{
	for (int i = 0; i < n; i++)
	{
		std::cout << i << " -> ";
		for (auto &[v, w] : adj[i])
		{
			std::cout << "(" << v << ", " << w << ") ";
		}
		std::cout << std::endl;
	}
}

void Graph::add_bi_edge(int u, int v, double w)
{
	assert(! (u < 0 || u >= n || v < 0 || v >= n || w <= 0));

	double neg_inf = std::numeric_limits<double>::min();
	
	// Check if edge is already in graph, remove if so
	auto u_to_v = adj[u].lower_bound(std::make_pair(v, neg_inf));
	if (u_to_v != adj[u].end() and u_to_v->first == v)
	{
		adj[u].erase(*u_to_v);
		m--;
	}
		
	auto v_to_u = adj[v].lower_bound(std::make_pair(u, neg_inf));
	if (v_to_u != adj[v].end() and v_to_u->first == u)
	{
		adj[v].erase(*v_to_u);
		m--;
	}	

	// Add edge with new_weight
	adj[u].insert(std::make_pair(v, w));
	adj[v].insert(std::make_pair(u, w));
	m += 2;
}

Graph* Graph::random_complete_graph(int n_)
{
	Graph* g = new Graph(n_);

	for (int i = 0; i < n_; i++)
	{
		for (int j = 0; j < n_; j++)
		{
			if (i == j)
				continue;
			double w = (rand() % 10) + 0.5;
			g->adj[i].insert(std::make_pair(j, w));
		}
	}

	return g;
}

Graph* Graph::prim()
{
	Graph* mst = new Graph(n);

	// Define infinity
	constexpr double inf = std::numeric_limits<double>::infinity();

	// Declare priority queue
	auto cmp = [](edge e1, edge e2) { return e1.second > e2.second; };
	std::priority_queue<edge, std::vector<edge>, decltype(cmp)> q(cmp);

	// Auxiliary vectors
	std::vector<double> key(n, inf);
	std::vector<int> parent(n, -1);
	std::vector<bool> inMST(n, false);

	// Prim algorithm
	q.push(std::make_pair(0, 0));
	key[0] = 0;

	while (!q.empty())
	{
		int u = q.top().first;
		q.pop();

		if (inMST[u])
			continue;
		inMST[u] = true;

		for (auto& [v, w] : adj[u])
		{
			if (!inMST[v] && key[v] > w)
			{
				key[v] = w;
				q.push(std::make_pair(v, w));
				parent[v] = u;
			}
		}
	}

	for (int i = 1; i < n; i++)
		mst->add_bi_edge(i, parent[i], key[i]);

	return mst;
}

Graph* Graph::kruskal(){
	Graph* mst = new Graph(n);
	
	std::vector<bi_edge> edge_list;

	for(int i = 0; i < n; i++)
		for(auto &[v, w] : adj[i])
			edge_list.push_back({w, v, i});
		
	sort(edge_list.begin(), edge_list.end());

	UnionFind components(n);

	for(auto &[w, u, v] : edge_list){
		if(components.isSameClass(u, v)) continue;
		
		mst->add_bi_edge(u, v, w);
		components.unionClass(u, v);
		if(components.getNumSets() == 1) break;
	}
	
	return mst;
}

/**
* Boruvska's algorithm to calculate MST.
* 
* @return: pointer to MST
*/
Graph* Graph::boruvska()
{
	Graph* mst = new Graph(n);

	std::vector<bi_edge> cheapest(n, std::make_tuple(-1, -1, -1));
	
	std::vector<bi_edge> edge_list;
	for (int i = 0; i < n; i++)
		for (auto& [v, w] : adj[i])
			edge_list.push_back({ w, v, i });

	UnionFind components(n);

	bool completed = false;
	while (!completed)
	{
		for (auto& [w, u, v] : edge_list)
		{
			int set1 = components.find(u);
			int set2 = components.find(v);

			if (set1 != set2)
			{
				if (std::get<1>(cheapest[set1]) == -1 or std::get<0>(cheapest[set1]) > w)
					cheapest[set1] = std::make_tuple(w, u, v);
				if (std::get<1>(cheapest[set2]) == -1 or std::get<0>(cheapest[set2]) > w)
					cheapest[set2] = std::make_tuple(w, u, v);
			}
		}

		for (int i = 0; i < n; i++)
		{
			if (std::get<1>(cheapest[i]) != -1)
			{
				auto& [w, u, v] = cheapest[i];
				int set1 = components.find(u);
				int set2 = components.find(v);

				if (set1 != set2)
				{
					components.unionClass(set1, set2);
					mst->add_bi_edge(u, v, w);
					if (components.getNumSets() == 1) completed = true;
				}
			}
		}
		cheapest = std::vector<bi_edge>(n, std::make_tuple(-1, -1, -1));
	}

	return mst;
}

/**
* Getter of adjacency list.
* 
* @return: adjacency list of graph
*/
std::set<edge>* Graph::get_adj()
{
	return adj;
}

/**
* Uses Kruskal's algorithm to calculate MST,
* then removes k - 1 heaviest edges in order
* to have k clusters.
* 
* @param: number of clusters k
* @return: cluster of each vertex
*/
std::vector<int> Graph::mst_cluster(int k)
{
	// Check if valid k
	assert(1 <= k and k <= n);

	std::vector<int> res(n, -1);

	// Get MST from Kruskal's
	Graph* mst = kruskal();
	std::set<edge>* mst_adj = mst->get_adj();

	// Get and sort bi_edges by weight
	std::vector<bi_edge> edge_list;
	for (int i = 0; i < n; i++)
		for (auto& [v, w] : mst_adj[i])
			edge_list.push_back({ w, v, i });
	sort(edge_list.begin(), edge_list.end());

	// Assign clusters
	UnionFind components(n);
	for (int i = 0; i < edge_list.size() - 2 * k + 2; i++)
	{
		auto& [w, u, v] = edge_list[i];
		if (!components.isSameClass(u, v))
			components.unionClass(u, v);
	}

	// Fill and return result
	for (int i = 0; i < n; i++)
		res[i] = components.find(i);

	delete mst;

	return res;
}

/**
* Finds optimal number of clusters k through
* silhouette method and applies mst_clustering(int k)
* to get optimal clustering
* 
* @return: cluster of each vertex
*/
std::vector<int> Graph::mst_cluster()
{
	std::vector<double> weights;
	for (int k = 1; k <= (int)std::log10(n); k++)
	{
		Graph* mst = kruskal();
		std::vector<int> clusters = mst_cluster(k);

		double total_w = 0;
		std::vector<bool> checked(n, false);
		for (int i = 0; i < n; i++)
		{
			for(auto &[v, w] : mst->adj[i])
				if (clusters[v] == clusters[i] && !checked[v])
				{
					checked[v] = true;
					total_w += w;
				}
		}

		weights.push_back(total_w);

		delete mst;
	}

	auto it = std::min_element(std::begin(weights), std::end(weights));
	int k_opt = 1 + std::distance(std::begin(weights), it);
	
	return mst_cluster(k_opt);
}