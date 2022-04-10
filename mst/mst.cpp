// mst.cpp : Defines the entry point for the application.
//

#include "mst.h"

int main()
{
	Graph g(5);

	try
	{
		g.add_bi_edge(0, 3, 1.3);
		g.add_bi_edge(0, 3, 42);
	}
	catch (std::invalid_argument& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
