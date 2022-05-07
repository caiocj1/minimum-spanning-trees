// mst.cpp : Defines the entry point for the application.
//

#include "mst.h"

int main()
{
	Graph g(5);

	g.add_bi_edge(0, 1, 1.2);
	g.add_bi_edge(0, 2, 2.4);
	g.add_bi_edge(0, 3, 3.6);
	g.add_bi_edge(1, 2, 1.1);
	g.add_bi_edge(2, 3, 2.2);
	g.add_bi_edge(1, 4, 3.3);
	g.add_bi_edge(2, 4, 4.4);
	g.add_bi_edge(3, 4, 5.5);

	/*Graph g(3);

	g.add_bi_edge(0, 1, 3.1);
	g.add_bi_edge(0, 2, 5.2);
	g.add_bi_edge(1, 2, 4.7);*/

	std::cout << "Graph g:" << std::endl;
	g.print();
	std::cout << std::endl;

	Graph* test = g.prim();
	std::cout << "MST:" << std::endl;
	test->print();
	std::cout << std::endl;

	std::cout << "Clusters:" << std::endl;
	std::vector<int> clusters = test->mst_clustering(2);
	for (int i = 0; i < g.get_n(); i++)
		std::cout << clusters[i] << " ";
	std::cout << std::endl;

	delete test;

	return 0;
}
