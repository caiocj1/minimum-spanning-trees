﻿// mst.cpp : Defines the entry point for the application.
//

#include "mst.h"

void task1()
{
	Graph* g = Graph::random_complete_graph(500);

	auto begin = std::chrono::steady_clock::now();
	Graph* mst = g->prim();
	auto end = std::chrono::steady_clock::now();
	auto dif = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "Prim time    : " << dif << " [ms]" << std::endl;

	delete mst;

	begin = std::chrono::steady_clock::now();
	mst = g->boruvska();
	end = std::chrono::steady_clock::now();
	dif = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "Boruvska time: " << dif << " [ms]" << std::endl;

	delete mst;

	delete g;
}

void task5()
{
	// Imports sample dataset that consists in 400 points
	// taken from two gaussian distributions
	Dataset compare("../data/compare.csv");
	
	// Measure time of k-means
	auto begin = std::chrono::steady_clock::now();

	// Performs k-means with Forgy initialization
	std::vector<int> labels_kmeans = compare.k_means(2);

	auto end = std::chrono::steady_clock::now();
	auto dif = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

	std::cout << "k-means time: " << dif << " [microseg]" << std::endl;

	// Creates complete graph from same dataset
	Graph g(compare);

	// Measures time for MST clustering
	begin = std::chrono::steady_clock::now();
	
	// Performs MST clustering
	std::vector<int> labels_mst = g.mst_cluster(2);

	end = std::chrono::steady_clock::now();
	dif = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	std::cout << "MST time    : " << dif << " [ms]" << std::endl;
}

void task6()
{
	Dataset test("../data/rio_airbnb_listings.csv");
	
	std::cout << test.get_n() << " " << test.get_d();

	//Graph g(test, "standardize");
}



int main()
{
	/*Graph g(5);

	g.add_bi_edge(0, 1, 1.2);
	g.add_bi_edge(0, 2, 2.4);
	g.add_bi_edge(0, 3, 3.6);
	g.add_bi_edge(1, 2, 1.1);
	g.add_bi_edge(2, 3, 2.2);
	g.add_bi_edge(1, 4, 3.3);
	g.add_bi_edge(2, 4, 4.4);
	g.add_bi_edge(3, 4, 5.5);

	Graph g(3);

	g.add_bi_edge(0, 1, 3.1);
	g.add_bi_edge(0, 2, 5.2);
	g.add_bi_edge(1, 2, 4.7);

	std::cout << "Graph g:" << std::endl;
	g.print();
	std::cout << std::endl;

	Graph* test = g.prim();
	std::cout << "MST:" << std::endl;
	test->print();
	std::cout << std::endl;

	std::cout << "Clusters:" << std::endl;
	std::vector<int> clusters = test->mst_cluster(2);
	for (int i = 0; i < g.get_n(); i++)
		std::cout << clusters[i] << " ";
	std::cout << std::endl;

	delete test;*/

	bool valid;
	do
	{
		std::cout << "Enter task number (1-6): " << std::flush;
		int task;
		std::cin >> task;
		valid = true;
		switch (task)
		{
		case 1:
			task1();
			break;
		case 5:
			task5();
			break;
		case 6:
			task6();
			break;
		default:
			valid = false;
			std::cout << "Invalid task number, try again." << std::endl;
			break;
		}
	} while (!valid);
	

	return 0;
}

