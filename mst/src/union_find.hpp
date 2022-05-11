#include <vector>


class UnionFind {
private:
    std::vector<int> parent, rank, setSize;
    int numSets;

public:
    UnionFind(int N){
        parent.assign(N, 0);
        for(int i = 0; i < N; i++)
            parent[i] = i;    // every element is its own class rep

        rank.assign(N, 0);    // every element is at the bottom
        setSize.assign(N, 1); // every class is a singleton
        numSets = N;          // we start with N classes
    }

    int find(int i) {
        return (parent[i] == i) ? i : (parent[i] = find(parent[i]));
    }

    bool isSameClass(int i, int j) { return find(i) == find(j);}
    int getNumSets() { return numSets; }
    int sizeOfClass(int i) { return setSize[find(i)];}

    void unionClass(int i, int j){
        if(isSameClass(i, j)) return;

        int x = find(i), y = find(j);
        if(rank[x] > rank[y]) std::swap(x, y);
        parent[x] = y;
        if(rank[x] == rank[y]) rank[y]++;
        setSize[y] += setSize[x];
        numSets--;
    }

};