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
            int n = argc > 1 ? atoi(argv[1]) : 400, m = n * (rand() % n);
            Graph *graph = Graph::random_sparse_graph(n, m), *mst;
            std::cout << "Running with " << size << " processes" << std::endl;
            std::cout << "Original graph: (n, m) = (" << n << ", " << m/2 << ") " << std::endl;
            //graph->print(); 
            ////////////////////////////
            double start = MPI_Wtime();
            mst = graph->prim();
            double time = MPI_Wtime() - start;
            
            std::cout << "MST found by sequential Prim: (n, m) = (" << mst->get_n() << ", " << mst->get_m()/2 << ") " << std::endl;
            //mst->print(); 
            delete mst;
            std::cout << "Time elapsed: " << 1000*time << " ms" << std::endl;
            ////////////////////////////
            start = MPI_Wtime();
            mst = graph->master_parallel_prim();
            time = MPI_Wtime() - start;
            
            std::cout << "MST found by parallel Prim: (n, m) = (" << mst->get_n() << ", " << mst->get_m()/2 << ") " << std::endl;
            //mst->print(); 
            delete mst;
            std::cout << "Time elapsed: " << 1000*time << " ms" << std::endl;
            ////////////////////////////
            start = MPI_Wtime();
            mst = graph->boruvska();
            time = MPI_Wtime() - start;
            
            std::cout << "MST found by sequential Boruvka: (n, m) = (" << mst->get_n() << ", " << mst->get_m()/2 << ") " << std::endl;
            //mst->print(); 
            delete mst;
            std::cout << "Time elapsed: " << 1000*time << " ms" << std::endl;
            ////////////////////////////
            start = MPI_Wtime();
            mst = graph->master_parallel_boruvska();
            time = MPI_Wtime() - start;
            
            std::cout << "MST found by parallel Boruvka: (n, m) = (" << mst->get_n() << ", " << mst->get_m()/2 << ") " << std::endl;
            //mst->print(); 
            delete mst;
            std::cout << "Time elapsed: " << 1000*time << " ms" << std::endl;
            ////////////////////////////
            delete graph; 
            break;
        }
        default:
        {
            Graph::slave_parallel_prim();
            Graph::slave_parallel_boruvska();
        }
    }

	MPI_Finalize();

	return EXIT_SUCCESS;
}
