#include "mst.h"
#include "src/graph.h"
#include "src/graph.cpp"
#include "src/dataset.cpp"
#include "src/dataset.hpp"
#include <mpi.h>

// main program
int main(int argc, char* argv[]) {
	// MPI variables and initialization
	int rank;
	int size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    constexpr int MASTER = 0;

    switch(rank){
        case 0:
        {
            int n = 4 + 4*(rand() % 7), m = n * (rand() % n);
            Graph *graph = Graph::random_sparse_graph(n, m), *mst;
            
            std::cout << "Original graph: (n, m) = (" << n << ", " << m/2 << ") " << std::endl;
            graph->print();

            double start = MPI_Wtime();
            mst = graph->master_parallel_boruvska();
            double time = MPI_Wtime() - start;
            
            std::cout << "MST found: (n, m) = (" << mst->get_n() << ", " << mst->get_m()/2 << ") " << std::endl;
            mst->print();
            std::cout << "Time elapsed: " << 1000*time << " ms" << std::endl;
            
            delete graph; delete mst;
            break;
        }
        default:
        {
            Graph::slave_parallel_boruvska();
        }
    }

	MPI_Finalize();

	return EXIT_SUCCESS;
}
