// mst.cpp : Defines the entry point for the application.
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
	// Imports sample dataset that consists in points
	// taken from three gaussian distributions.
	// Change to 600, 900, 1200, 1500 to get time dependance
	Dataset compare("../data/compare_3clusters_300_separated.csv");
	
	// Measure time of k-means
	auto begin = std::chrono::steady_clock::now();

	// Performs k-means with Forgy initialization
	std::vector<int> labels_kmeans = compare.k_means(3);

	auto end = std::chrono::steady_clock::now();
	auto dif = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

	std::cout << "k-means time: " << dif << " [microseg]" << std::endl;

	// Creates complete graph from same dataset
	Graph g(compare);

	// Measures time for MST clustering
	begin = std::chrono::steady_clock::now();
	
	// Performs MST clustering
	std::vector<int> labels_mst = g.mst_cluster(3);

	end = std::chrono::steady_clock::now();
	dif = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	std::cout << "MST time    : " << dif << " [ms]" << std::endl;

	for (int i = 0; i < g.get_n(); i++)
		std::cout << labels_mst[i] << " ";
}

void task6()
{
	Dataset test("../data/compare_3clusters_300_separated.csv");
	
	std::cout << test.get_n() << " " << test.get_d() << std::endl;

	Graph g(test, "standardize");

	// Measure time of mst_clusters with no k
	auto begin = std::chrono::steady_clock::now();

	std::vector<int> clusters = g.mst_cluster();

	auto end = std::chrono::steady_clock::now();
	auto dif = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	std::cout << "MST cluster (no explicit k) time: " << dif << " [ms]" << std::endl;
}



int main()
{
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

