// mst.cpp : Defines the entry point for the application.
//

#include "mst.h"

int main()
{
	Graph g(5);

	g.add_bi_edge(0, 1, 2);
	g.add_bi_edge(0, 2, 1);
	g.add_bi_edge(0, 3, 2);
	g.add_bi_edge(1, 3, 3);
	g.add_bi_edge(1, 4, 4);
	g.add_bi_edge(2, 3, 1);
	g.add_bi_edge(3, 4, 5);
	std::cout << "Graph g:" << std::endl;
	g.print();
	std::cout << std::endl;

	Graph* test = g.prim();
	std::cout << "MST:" << std::endl;
	test->print();
	delete test;

	return 0;
}
