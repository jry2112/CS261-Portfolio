# Course: CS261 - Data Structures
# Author: Jada Young
# Assignment: Assignment 6 - Directed Graph
# Description: Implementation of Directed Graph with adjacency matrix. Can support the following types of graphs:
# directed, weighted (positive edge weights only), no duplicate edges, no loops. Cycles are allowed.
# Includes the following methods: 1) add_vertex(), add_edge(); 2) remove_edge(), get_vertices(), get_edges(); 4)
# is_valid_path(), dfs(), bfs(); 5) has_cycles(), dijkstra()

import heapq
from collections import deque

class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        Adds a new vertex to the graph. Vertex name does not need to be provided; vertex is assigned a reference
        index (integer): 0, 1, 2, 3, ..., n. Returns a single integer - the number of vertices in the graph after the
        addition.
        """
        # base case - adj matrix is empty
        if len(self.adj_matrix) == 0:
            self.adj_matrix.append([0])
        else:
            index = self.v_count
            #increase x
            for row in self.adj_matrix:
                row.append(0)
            # increase y - add a new list to the end
            self.adj_matrix.append([])
            # fill new row with 0 since edges haven't been added yet
            self.adj_matrix[index] = [0 for _ in self.adj_matrix[0]]
        self.v_count += 1
        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Adds a new edge to the graph, connecting two vertices with provided indices. If either (or both) vertex
        indices do not exist in the graph, or if the weight is not a positive integer, or if src and dst refer to the
        same vertex, does nothing. If an edge already exists, updates weight.
        """
        # valid indices
        if 0 < src < self.v_count and 0 < dst < self.v_count and src != dst:
            # valid weight
            if weight > 0:
                # i-th row, j-th col = weight of i -> j
                row = self.adj_matrix[src]
                row[dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Removes an edge between two given vertices. If either (or both) vertex indices do not exist or if there is no
        edge between them, does nothing.
        """
        # validate indices
        if 0 < src < self.v_count and 0 < dst < self.v_count and src != dst:
            # src -> row
            row = self.adj_matrix[src]
            row[dst] = 0

    def get_vertices(self) -> []:
        """
        Returns a list of vertices.
        """
        # vertices = count - 1
        vertices = [i for i in range(self.v_count)]
        return vertices

    def get_edges(self) -> []:
        """
        Returns list of edges in the graph in the form (src, dst, weight).
        """
        edges = []
        # row -> src, col -> dst, [row][col] -> weight
        for row in range(self.v_count):
            for col in range(self.v_count):
                weight = self.adj_matrix[row][col]
                if weight > 0:
                    edges.append((row, col, weight))
        return edges

    def is_valid_path(self, path: []) -> bool:
        """
        TODO: Write this implementation
        """
        pass

    def dfs(self, v_start, v_end=None) -> []:
        """
        TODO: Write this implementation
        """
        pass

    def bfs(self, v_start, v_end=None) -> []:
        """
        TODO: Write this implementation
        """
        pass

    def has_cycle(self):
        """
        TODO: Write this implementation
        """
        pass

    def dijkstra(self, src: int) -> []:
        """
        TODO: Write this implementation
        """
        pass


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)


    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)


    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
