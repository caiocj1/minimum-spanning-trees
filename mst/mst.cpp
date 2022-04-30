// mst.cpp : Defines the entry point for the application.
//

#include "mst.h"

int main()
{
	Graph g(3);

	g.add_bi_edge(0, 1, 2.5);
	g.add_bi_edge(0, 1, 76.5);
	g.add_bi_edge(1, 2, 10.5);
	g.add_bi_edge(0, 2, 5.5);
	g.add_bi_edge(1, 2, 90.5);
	std::cout << "Graph g:" << std::endl;
	g.print();


	std::cout << std::endl;

	Graph* test = g.kruskal();
	std::cout << "MST:" << std::endl;
	test->print();

	delete test;

	return 0;
}
