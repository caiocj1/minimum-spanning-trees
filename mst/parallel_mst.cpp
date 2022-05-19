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
            Graph *graph = new Graph(6), *mst;
            graph->add_bi_edge(0, 1, 1);
            graph->add_bi_edge(0, 2, 3);
            graph->add_bi_edge(0, 5, 3);
            graph->add_bi_edge(1, 2, 5);
            graph->add_bi_edge(1, 3, 1);
            graph->add_bi_edge(2, 3, 2);
            graph->add_bi_edge(2, 4, 1);
            graph->add_bi_edge(3, 4, 4);
            graph->add_bi_edge(4, 5, 5);
            //double start = MPI_Wtime();
            mst = graph->master_parallel_boruvska();
            
            mst->print();
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
