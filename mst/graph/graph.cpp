#include "graph.h"
#include "union_find.hpp"
#include <stdexcept>
#include <iostream>
#include <queue>
#include <limits>

Graph::Graph(int n_)
{
	n = n_;
	m = 0;
	adj = new std::list<edge>[n_];
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
		for (auto p = adj[i].begin(); p != adj[i].end(); p++)
		{
			std::cout << "(" << p->first << ", " << p->second << ") ";
		}
		std::cout << std::endl;
	}
}

void Graph::add_bi_edge(int u, int v, double w)
{
	if (u < 0 || u >= n || v < 0 || v >= n || w <= 0)
		throw std::invalid_argument(
			"Vertices must be in [0, n[ and weight must be strictly positive");

	for (auto p = adj[u].begin(); p != adj[u].end(); p++)
		if (p->first == v)
		{
			p->second = w;
			for (auto q = adj[v].begin(); q != adj[v].end(); q++)
				if (q->first == u)
				{
					q->second = w;
					return;
				}
		}

	adj[u].push_back(std::make_pair(v, w));
	adj[v].push_back(std::make_pair(u, w));
	m++;
}

Graph* Graph::prim()
{
	Graph* mst = new Graph(n);

	double inf = std::numeric_limits<double>::infinity();

	auto cmp = [](edge e1, edge e2) { return e1.second > e2.second; };
	std::priority_queue<edge, std::vector<edge>, decltype(cmp)> q(cmp);

	std::vector<double> key(n, inf);
	std::vector<int> parent(n, -1);
	std::vector<bool> inMST(n, false);

	q.push(std::make_pair(0, 0));
	key[0] = 0;

	while (!q.empty())
	{
		int u = q.top().first;
		q.pop();

		if (inMST[u])
			continue;

		inMST[u] = true;

		for (auto p = adj[u].begin(); p != adj[u].end(); p++)
		{
			int v = p->first;
			double w = p->second;

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
	
}