#include "graph.h"
#include "union_find.hpp"
#include <stdexcept>
#include <iostream>
#include <queue>
#include <limits>
#include <random>
#include <algorithm>
#include <assert.h>

Graph::Graph(int n_)
{
	n = n_;
	m = 0;
	adj = new std::set<edge>[n_];
}

Graph::Graph(Dataset data, std::string normalize)
{
	n = data.get_n();
	adj = new std::set<edge>[n];
	m = 0;

	if (normalize == "standardize")
		data.standardize();
	else if (normalize == "rescale")
		data.rescale();

	for (int i = 0; i < n; i++)
	{
		std::vector<double> x_i = data.get_instance(i);
		for (int j = i + 1; j < n; j++)
		{
			std::vector<double> x_j = data.get_instance(j);
			add_bi_edge(i, j, Dataset::dist(x_i, x_j));
		}
	}
}

Graph::~Graph()
{
	delete[] adj;
}

int Graph::get_n()
{
	return n;
}

int Graph::get_m()
{
	return m;
}

void Graph::print()
{
	for (int i = 0; i < n; i++)
	{
		std::cout << i << " -> ";
		for (auto &[v, w] : adj[i])
		{
			std::cout << "(" << v << ", " << w << ") ";
		}
		std::cout << std::endl;
	}
}

void Graph::add_bi_edge(int u, int v, double w)
{
	assert(! (u < 0 || u >= n || v < 0 || v >= n || w <= 0));

	double neg_inf = std::numeric_limits<double>::min();
	
	// Check if edge is already in graph, remove if so
	auto u_to_v = adj[u].lower_bound(std::make_pair(v, neg_inf));
	if (u_to_v != adj[u].end() and u_to_v->first == v)
	{
		adj[u].erase(*u_to_v);
		m--;
	}
		
	auto v_to_u = adj[v].lower_bound(std::make_pair(u, neg_inf));
	if (v_to_u != adj[v].end() and v_to_u->first == u)
	{
		adj[v].erase(*v_to_u);
		m--;
	}	

	// Add edge with new_weight
	adj[u].insert(std::make_pair(v, w));
	adj[v].insert(std::make_pair(u, w));
	m += 2;
}

Graph* Graph::random_complete_graph(int n_)
{
	Graph* g = new Graph(n_);

	for (int i = 0; i < n_; i++)
	{
		for (int j = 0; j < n_; j++)
		{
			if (i == j)
				continue;
			double w = (rand() % 10) + 0.5;
			g->add_bi_edge(i, j, w);
		}
	}

	return g;
}

Graph* Graph::random_sparse_graph(int n_, int m_){
	Graph* g = new Graph(n_);
	UnionFind components(n_);

	std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(n_, 2*n_);
	
	while(components.getNumSets() > 1){
		int u = rand() % n_, v = rand() % n_;
		while(components.isSameClass(u, v)) u = rand() % n_, v = rand() % n_;
		g->add_bi_edge(u, v, dist(e2));
		components.unionClass(u, v);
	}
	while(g->m < m_){
		int u = rand() % n_, v = rand() % n_;
		while(u == v) u = rand() % n_, v = rand() % n_;
		g->add_bi_edge(u, v, dist(e2));
		components.unionClass(u, v);
	}

	return g;
}

/**
* Prim's algorithm to calculate MST.
* 
* @return: pointer to MST
*/
Graph* Graph::prim()
{
	Graph* mst = new Graph(n);

	// Define infinity
	constexpr double inf = std::numeric_limits<double>::infinity();

	// Declare priority queue
	auto cmp = [](edge e1, edge e2) { return e1.second > e2.second; };
	std::priority_queue<edge, std::vector<edge>, decltype(cmp)> q(cmp);

	// Auxiliary vectors
	std::vector<double> key(n, inf);
	std::vector<int> parent(n, -1);
	std::vector<bool> inMST(n, false);

	// Prim algorithm
	q.push(std::make_pair(0, 0));
	key[0] = 0;

	while (!q.empty())
	{
		int u = q.top().first;
		q.pop();

		if (inMST[u])
			continue;
		inMST[u] = true;

		for (auto& [v, w] : adj[u])
		{
			if (!inMST[v] && key[v] > w)
			{
				key[v] = w;
				q.push(std::make_pair(v, w));
				parent[v] = u;
			}
		}
	}

	for (int i = 1; i < n; i++)
		mst->add_bi_edge(i, parent[i], key[i]);

	return mst;
}

/**
* Master process version of the parallelized Prim's algorithm.
* 
* @return: pointer to MST
*/
Graph* Graph::master_parallel_prim(){
	const int MASTER = 0;
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    if(p < 2) return prim();
    
    Graph *mst = new Graph(n);
    
    // Define infinity
	constexpr double inf = std::numeric_limits<double>::infinity();

	// Declare priority queue
	auto cmp = [](edge e1, edge e2) { return e1.second > e2.second; };
	std::priority_queue<edge, std::vector<edge>, decltype(cmp)> q(cmp);

    // Distribute vertices among processes
    int m_load = n/p + n%p, s_load = n/p;
    std::vector<int> buf = {m_load, s_load};
    
    MPI_Bcast(buf.data(), 2, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Auxiliary function
    auto is_vertex_mine = [m_load, s_load](int v) {return v < m_load;};

	// Auxiliary vectors
	std::vector<double> key(m_load, inf);
	std::vector<int> parent(m_load, -1);
	std::vector<bool> inMST(m_load, false);
	
    // Prim algorithm: 
    // Initialize by setting the root
	q.push(std::make_pair(0, 0));
	key[0] = 0;

	// Algorithm's main loop: exactly n vertices must be added to the MST
    for(int i = 0, u, num_edges; i < n; i++){
        std::pair<double, int> mwoe;
        while(!q.empty() && inMST[q.top().first]) q.pop();
        mwoe = q.empty() ? std::make_pair(inf, -1) : 
                            std::make_pair(q.top().second, q.top().first);
        // Send out the local MWOE for scrutiny                            
        MPI_Allreduce(&mwoe, &mwoe, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        u = mwoe.second;
        // If the global MWOE is mine, handle it
        if(is_vertex_mine(u))
            inMST[u] = true;
        
        // Master sends everyone the adj list of the new node
        num_edges = adj[u].size();
        MPI_Bcast(&num_edges, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

        std::vector<std::pair<double, int> > neighbors;
        for(auto &[v, w] : adj[u]) neighbors.push_back(std::make_pair(w, v));
        MPI_Bcast(neighbors.data(), num_edges, MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);

        // Every process incorporates the new node's edges info
        for(auto &[w, v] : neighbors){
            if(is_vertex_mine(v) && !inMST[v] && key[v] > w){
                key[v] = w;
                q.push(std::make_pair(v, w));
                parent[v] = u;
            }
        }
    }

    std::vector<double> global_key(n);
	std::vector<int> global_parent(n);

    std::vector<int> recvcounts(p), displs(p);
    for(int r = 0; r < p; r++) 
        (recvcounts[r] = r ? s_load : m_load, displs[r] =  r ? displs[r-1] + recvcounts[r-1] : 0);
    
    MPI_Gatherv(key.data(), m_load, MPI_DOUBLE, 
                global_key.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                MASTER, MPI_COMM_WORLD);
    MPI_Gatherv(parent.data(), m_load, MPI_INT, 
                global_parent.data(), recvcounts.data(), displs.data(), MPI_INT, 
                MASTER, MPI_COMM_WORLD);

	for (int i = 1; i < n; i++)
        mst->add_bi_edge(i, global_parent[i], global_key[i]);

	return mst;
}

/**
* Slave processes version of the parallelized Prim's algorithm.
* 
* @return: pointer to MST
*/
void Graph::slave_parallel_prim(){
	const int MASTER = 0;
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Status status;
    
    // Define infinity
	constexpr double inf = std::numeric_limits<double>::infinity();

	// Declare priority queue
	auto cmp = [](edge e1, edge e2) { return e1.second > e2.second; };
	std::priority_queue<edge, std::vector<edge>, decltype(cmp)> q(cmp);

    std::vector<int> buf(2);
    MPI_Bcast(buf.data(), 2, MPI_INT, MASTER, MPI_COMM_WORLD);
    int m_load = buf[0], s_load = buf[1], n = m_load + (p-1)*s_load;
        
	// Auxiliary vectors
	std::vector<double> key(s_load, inf);
	std::vector<int> parent(s_load, -1);
	std::vector<bool> inMST(s_load, false);

    // Auxiliary functions
    auto is_vertex_mine = [m_load, rank, s_load](int v) {
        return (m_load + (rank-1)*s_load <= v) && (v < m_load + rank*s_load);
    };
    auto vertex_to_index = [m_load, rank, s_load](int v) {return v - m_load - (rank-1)*s_load;};
    auto index_to_vertex = [m_load, rank, s_load](int i) {return m_load + (rank-1)*s_load + i;};
    
    // Adjacency list of the newest node in the MST
    std::set<edge> newest_adj;

    // Prim algorithm: 
    for(int i = 0, u, num_edges; i < n; i++){
        std::pair<double, int> mwoe;
        while(!q.empty() && inMST[vertex_to_index(q.top().first)]) q.pop();
        mwoe = q.empty() ? std::make_pair(inf, -1) : 
                            std::make_pair(q.top().second, q.top().first);
        // Send out the local MWOE for scrutiny                            
        MPI_Allreduce(&mwoe, &mwoe, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        u = mwoe.second;
        // If the global MWOE is mine, handle it
        if(is_vertex_mine(u))
            inMST[vertex_to_index(u)] = true;
        
        // Master sends everyone the adj list of the new node
        MPI_Bcast(&num_edges, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

        std::vector<std::pair<double, int> > neighbors(num_edges);
        MPI_Bcast(neighbors.data(), num_edges, MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);

        // Every process incorporates the new node's edges info
        for(auto &[w, v] : neighbors){
            if(is_vertex_mine(v) && !inMST[vertex_to_index(v)] && key[vertex_to_index(v)] > w){
                key[vertex_to_index(v)] = w;
                q.push(std::make_pair(v, w));
                parent[vertex_to_index(v)] = u;
            }
        }
    }

    MPI_Gatherv(key.data(), s_load, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Gatherv(parent.data(), s_load, MPI_INT, NULL, NULL, NULL, MPI_INT, MASTER, MPI_COMM_WORLD);
}

/**
* Kruskal's algorithm to calculate MST.
* 
* @return: pointer to MST
*/
Graph* Graph::kruskal(){
	Graph* mst = new Graph(n);
	
	std::vector<bi_edge> edge_list;

	for(int i = 0; i < n; i++)
		for(auto &[v, w] : adj[i])
			edge_list.push_back({w, v, i});
		
	std::sort(edge_list.begin(), edge_list.end());

	UnionFind components(n);

	for(auto &[w, u, v] : edge_list){
		if(components.isSameClass(u, v)) continue;
		
		mst->add_bi_edge(u, v, w);
		components.unionClass(u, v);
		if(components.getNumSets() == 1) break;
	}
	
	return mst;
}

/**
* Boruvska's algorithm to calculate MST.
* 
* @return: pointer to MST
*/
Graph* Graph::boruvska()
{
	Graph* mst = new Graph(n);

	std::vector<bi_edge> cheapest(n, std::make_tuple(-1, -1, -1));
	
	std::vector<bi_edge> edge_list;
	for (int i = 0; i < n; i++)
		for (auto& [v, w] : adj[i])
			edge_list.push_back({ w, v, i });

	UnionFind components(n);

	bool completed = false;
	while (!completed)
	{
		for (auto& [w, u, v] : edge_list)
		{
			int set1 = components.find(u);
			int set2 = components.find(v);

			if (set1 != set2)
			{
				if (std::get<1>(cheapest[set1]) == -1 or std::get<0>(cheapest[set1]) > w)
					cheapest[set1] = std::make_tuple(w, u, v);
				if (std::get<1>(cheapest[set2]) == -1 or std::get<0>(cheapest[set2]) > w)
					cheapest[set2] = std::make_tuple(w, u, v);
			}
		}

		for (int i = 0; i < n; i++)
		{
			if (std::get<1>(cheapest[i]) != -1)
			{
				auto& [w, u, v] = cheapest[i];
				int set1 = components.find(u);
				int set2 = components.find(v);

				if (set1 != set2)
				{
					components.unionClass(set1, set2);
					mst->add_bi_edge(u, v, w);
					if (components.getNumSets() == 1) completed = true;
				}
			}
		}
		cheapest = std::vector<bi_edge>(n, std::make_tuple(-1, -1, -1));
	}

	return mst;
}

/**
* Master process version of the parallelized Boruvka's algorithm.
* 
* @return: pointer to MST
*/
Graph* Graph::master_parallel_boruvska(){
    const int MASTER = 0;
    int p; MPI_Comm_size(MPI_COMM_WORLD, &p);
    if(p < 2) return boruvska();
    
    Graph *mst = new Graph(n);
    
	constexpr double inf = std::numeric_limits<double>::infinity();
	auto cmp = [](edge e1, edge e2) { return e1.second > e2.second; };

    // Distribute vertices among processes
    int m_load = n/p + n%p, s_load = n/p;
    std::vector<int> buf = {m_load, s_load};
    MPI_Bcast(buf.data(), 2, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Auxiliary function
    auto is_vertex_mine = [m_load, s_load](int v) {return v < m_load;};

    // Scatter edges throughout processes
    std::vector<int> sendbuf(n), recvbuf(m_load);
    for(int i = 0; i < n; i++) sendbuf[i] = adj[i].size();
    std::vector<int> sendcounts(p), displs(p);
    for(int r = 0; r < p; r++) 
        sendcounts[r] = r ? s_load : m_load, displs[r] = r ? displs[r-1] + sendcounts[r-1] : 0;
    
    MPI_Scatterv(sendbuf.data(), sendcounts.data(), displs.data(), MPI_INT, 
                 recvbuf.data(), m_load, MPI_INT, MASTER, MPI_COMM_WORLD);

    std::vector<edge> edge_sendbuf, 
                      edge_recvbuf(std::accumulate(recvbuf.begin(), recvbuf.end(), 0));
    for(int i = 0; i < n; i++) 
        for(edge e : adj[i]) 
            edge_sendbuf.push_back(e);

    for(int r = 0; r < p; r++) 
        sendcounts[r] = std::accumulate(sendbuf.begin()+displs[r], sendbuf.begin()+displs[r]+sendcounts[r], 0), 
        displs[r] = r ? displs[r-1] + sendcounts[r-1] : 0;

    MPI_Datatype mpi_edge; 
    struct {int bl[2]; MPI_Aint dsp[2]; MPI_Datatype tp[2];} ts;
    ts = {{1, 1}, {offsetof(edge, first), offsetof(edge, second)}, {MPI_INT, MPI_DOUBLE}};
    MPI_Type_create_struct(2, ts.bl, ts.dsp, ts.tp, &mpi_edge);
    MPI_Type_commit(&mpi_edge);
    
    MPI_Scatterv(edge_sendbuf.data(), sendcounts.data(), displs.data(), mpi_edge, 
                 edge_recvbuf.data(), edge_recvbuf.size(), mpi_edge, MASTER, MPI_COMM_WORLD);

    // Declare local array of adjacency lists
    std::vector<std::vector<edge> > local_adj(m_load);
    for(int i = 0, displs = 0; i < m_load; displs += recvbuf[i++]){
        local_adj[i] = std::vector<edge>(edge_recvbuf.begin()+displs, edge_recvbuf.begin()+displs+recvbuf[i]);
        std::sort(local_adj[i].begin(), local_adj[i].end(), cmp);
    }

    // Now every process has its local version of weight-sorted adjacency lists for each of its vertices

    // Begin Boruvska's algorithm: initialize UFDS of connected components
    UnionFind components(n);
    
    // build an array of MWOEs for each component
    std::vector<bi_edge> mwoes(n), mwoes_buf(n);
    
    // new MPI type
    MPI_Datatype mpi_bi_edge; 
    ts = {{1, 2}, {0, sizeof(double)}, {MPI_DOUBLE, MPI_INT}};
    MPI_Type_create_struct(2, ts.bl, ts.dsp, ts.tp, &mpi_bi_edge);
    MPI_Type_commit(&mpi_bi_edge);

    // new MPI operation to be used in MPI_Reduce()
    MPI_Op mpi_bi_edge_minloc;
    auto bi_edge_minloc = [] (void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
        auto in = static_cast<bi_edge *>(invec), inout = static_cast<bi_edge *>(inoutvec);
        for(int i = 0; i < *len; i++)
            if(std::get<0>(in[i]) < std::get<0>(inout[i]))
                inout[i] = in[i];
        return;
    };
    MPI_Op_create(bi_edge_minloc, true, &mpi_bi_edge_minloc);


    while(components.getNumSets() > 1){
        // Reset/fill the MWOE array with wildcard bi_edges
        mwoes.assign(n, std::make_tuple<double, int, int>((double) inf, -1, -1));
        mwoes_buf.assign(n, std::make_tuple<double, int, int>((double) inf, -1, -1));

        // Search for MWOEs in each of its vertices' adjacency lists
        for(int i = 0, u; i < local_adj.size(); i++){
            u = i; //slave version: u = index_to_vertex[i]
            // Find a candidate MWOE for the class of element u:

            while(!local_adj[i].empty() && (components.isSameClass(u, local_adj[i].back().first) || 
                  local_adj[i].back().second >= std::get<0>(mwoes[components.find(u)])))
                local_adj[i].pop_back();
            

            if(local_adj[i].empty()) continue;
            auto &[v, w] = local_adj[i].back();

            mwoes[components.find(u)] = std::make_tuple(w, u, v);
        }

        // All processes put together their candidate MWOE's, get the global ones
        MPI_Allreduce(mwoes.data(), mwoes_buf.data(), n, mpi_bi_edge, mpi_bi_edge_minloc, MPI_COMM_WORLD);
        mwoes.swap(mwoes_buf);
        // Update UFDS with the new mwoes
        for(auto &[w, u, v] : mwoes)
            if(w < inf && !components.isSameClass(u, v)) 
                mst->add_bi_edge(u, v, w), components.unionClass(u, v);

        mwoes.clear(); mwoes_buf.clear();
    }
    
    MPI_Type_free(&mpi_edge);
    MPI_Type_free(&mpi_bi_edge); 
    MPI_Op_free(&mpi_bi_edge_minloc);
    
	return mst;
}

/**
* Slave processes version of the parallelized Boruvka's algorithm.
* 
* @return: pointer to MST
*/
void Graph::slave_parallel_boruvska(){
    const int MASTER = 0;
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        
    // Define infinity
	constexpr double inf = std::numeric_limits<double>::infinity();
	auto cmp = [](edge e1, edge e2) { return e1.second > e2.second; };
	
    std::vector<int> buf(2);
    MPI_Bcast(buf.data(), 2, MPI_INT, MASTER, MPI_COMM_WORLD);
    int m_load = buf[0], s_load = buf[1], n = m_load + (p-1)*s_load;
        
	// Auxiliary functions
    auto is_vertex_mine = [m_load, rank, s_load](int v) {
        return (m_load + (rank-1)*s_load <= v) && (v < m_load + rank*s_load);
    };
    auto vertex_to_index = [m_load, rank, s_load](int v) {return v - m_load - (rank-1)*s_load;};
    auto index_to_vertex = [m_load, rank, s_load](int i) {return m_load + (rank-1)*s_load + i;};


    // Scatter edges throughout processes
    std::vector<int> recvbuf(s_load);
    
    MPI_Scatterv(NULL, NULL, NULL, MPI_INT, 
                 recvbuf.data(), s_load, MPI_INT, MASTER, MPI_COMM_WORLD);

    std::vector<edge> edge_recvbuf(std::accumulate(recvbuf.begin(), recvbuf.end(), 0));

    MPI_Datatype mpi_edge; 
    struct {int bl[2]; MPI_Aint dsp[2]; MPI_Datatype tp[2];} ts;
    ts = {{1, 1}, {offsetof(edge, first), offsetof(edge, second)}, {MPI_INT, MPI_DOUBLE}};
    MPI_Type_create_struct(2, ts.bl, ts.dsp, ts.tp, &mpi_edge);
    MPI_Type_commit(&mpi_edge);
    
    MPI_Scatterv(NULL, NULL, NULL, mpi_edge, 
                 edge_recvbuf.data(), edge_recvbuf.size(), mpi_edge, MASTER, MPI_COMM_WORLD);

    // Declare local array of adjacency lists
    std::vector<std::vector<edge> > local_adj(s_load);
    for(int i = 0, displs = 0; i < s_load; displs += recvbuf[i++]){
        local_adj[i] = std::vector<edge>(edge_recvbuf.begin()+displs, edge_recvbuf.begin()+displs+recvbuf[i]);
        std::sort(local_adj[i].begin(), local_adj[i].end(), cmp);
    }

    // Now every process has its local version of weight-sorted adjacency lists for each of its vertices

    // Begin Boruvska's algorithm: initialize UFDS of connected components
    UnionFind components(n);
    
    // build an array of MWOEs for each component
    std::vector<bi_edge> mwoes(n), mwoes_buf(n);
    
    // new MPI type
    MPI_Datatype mpi_bi_edge; 
    ts = {{1, 1}, {0, sizeof(double)}, {MPI_DOUBLE, MPI_2INT}};
    MPI_Type_create_struct(2, ts.bl, ts.dsp, ts.tp, &mpi_bi_edge);
    MPI_Type_commit(&mpi_bi_edge);

    // new MPI operation to be used in MPI_Reduce()
    MPI_Op mpi_bi_edge_minloc;
    auto bi_edge_minloc = [] (void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
        auto in = static_cast<bi_edge *>(invec), inout = static_cast<bi_edge *>(inoutvec);
        for(int i = 0; i < *len; i++)
            if(std::get<0>(in[i]) < std::get<0>(inout[i]))
                inout[i] = in[i];
        return;
    };
    MPI_Op_create(bi_edge_minloc, true, &mpi_bi_edge_minloc);


    while(components.getNumSets() > 1){
        // Reset/fill the MWOE array with wildcard bi_edges
        mwoes.assign(n, std::make_tuple<double, int, int>((double) inf, -1, -1));
        mwoes_buf.assign(n, std::make_tuple<double, int, int>((double) inf, -1, -1));

        // Search for MWOEs in each of its vertices' adjacency lists
        for(int i = 0, u; i < local_adj.size(); i++){
            u = index_to_vertex(i); 
            // Find a candidate MWOE for the class of element u:
            while(!local_adj[i].empty() && (components.isSameClass(u, local_adj[i].back().first) || 
                  local_adj[i].back().second >= std::get<0>(mwoes[components.find(u)])))
                local_adj[i].pop_back();
            
            if(local_adj[i].empty()) continue;
            auto &[v, w] = local_adj[i].back();
            mwoes[components.find(u)] = std::make_tuple(w, u, v);
        }
        
        // All processes put together their candidate MWOE's, reduce to get the global ones
        MPI_Allreduce(mwoes.data(), mwoes_buf.data(), n, mpi_bi_edge, mpi_bi_edge_minloc, MPI_COMM_WORLD);
        mwoes.swap(mwoes_buf);

        // Update UFDS with the new MWOEs
        for(auto &[w, u, v] : mwoes)
            if(w < inf && !components.isSameClass(u, v)) 
                components.unionClass(u, v);

    }

    MPI_Type_free(&mpi_edge);
    MPI_Type_free(&mpi_bi_edge); 
    MPI_Op_free(&mpi_bi_edge_minloc);
}

/**
* Getter of adjacency list.
* 
* @return: adjacency list of graph
*/
std::set<edge>* Graph::get_adj()
{
	return adj;
}

/**
* Uses Kruskal's algorithm to calculate MST,
* then removes k - 1 heaviest edges in order
* to have k clusters.
* 
* @param: number of clusters k
* @return: cluster of each vertex
*/
std::vector<int> Graph::mst_cluster(int k)
{
	// Check if valid k
	assert(1 <= k and k <= n);

	std::vector<int> res(n, -1);

	// Get MST from Kruskal's
	Graph* mst = kruskal();
	std::set<edge>* mst_adj = mst->get_adj();

	// Get and sort bi_edges by weight
	std::vector<bi_edge> edge_list;
	for (int i = 0; i < n; i++)
		for (auto& [v, w] : mst_adj[i])
			edge_list.push_back({ w, v, i });
	sort(edge_list.begin(), edge_list.end());

	// Assign clusters
	UnionFind components(n);
	for (int i = 0; i < edge_list.size() - 2 * k + 2; i++)
	{
		auto& [w, u, v] = edge_list[i];
		if (!components.isSameClass(u, v))
			components.unionClass(u, v);
	}

	// Fill and return result
	for (int i = 0; i < n; i++)
		res[i] = components.find(i);

	delete mst;

	return res;
}

/**
* Finds optimal number of clusters k through
* silhouette method and applies mst_clustering(int k)
* to get optimal clustering
* 
* @return: cluster of each vertex
*/
std::vector<int> Graph::mst_cluster()
{
	std::vector<double> weights;
	for (int k = 1; k <= (int)std::log10(n); k++)
	{
		Graph* mst = kruskal();
		std::vector<int> clusters = mst_cluster(k);

		double total_w = 0;
		std::vector<bool> checked(n, false);
		for (int i = 0; i < n; i++)
		{
			for(auto &[v, w] : mst->adj[i])
				if (clusters[v] == clusters[i] && !checked[v])
				{
					checked[v] = true;
					total_w += w;
				}
		}

		weights.push_back(total_w);

		delete mst;
	}

	auto it = std::min_element(std::begin(weights), std::end(weights));
	int k_opt = 1 + std::distance(std::begin(weights), it);
	
	return mst_cluster(k_opt);
}