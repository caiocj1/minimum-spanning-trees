#include "graph.h"
#include <stdexcept>
#include <iostream>

Graph::Graph(int n_)
{
	n = n_;
	adj = new std::list<std::pair<int,float>>[n_];
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
		std::cout << i << " connected to: ";
		for (auto p = adj[i].begin(); p != adj[i].end(); p++)
		{
			std::cout << "(" << p->first << ", " << p->second << ") ";
		}
		std::cout << std::endl;
	}
		
}

void Graph::add_bi_edge(int u, int v, float w)
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
					q->second = w;
			return;
		}

	adj[u].push_back(std::make_pair(v, w));
	adj[v].push_back(std::make_pair(u, w));
}