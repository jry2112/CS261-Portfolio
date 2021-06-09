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
        if 0 <= src < self.v_count and 0 <= dst < self.v_count and src != dst:
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
        if 0 <= src < self.v_count and 0 <= dst < self.v_count and src != dst:
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
        Takes a list of vertex indices and returns True if the sequence represents a valid path. Empty paths and
        single-vertex paths are considered valid.
        """
        valid_flag = True
        # base case - empty or single vertex:
        if len(path) == 0:
            return valid_flag
        elif len(path) == 1 and path[0] < self.v_count:
            return valid_flag
        else:
            i = 0
            src = path[i]
            # while we haven't reached the end of the path and path is still valid
            while valid_flag and i < len(path)-1:
                dest = path[i+1]
                # check if destination vertex is valid
                if self.adj_matrix[src][dest] == 0:
                    valid_flag = False
                else:
                    # keep looking
                    i += 1
                    src = path[i]
            return valid_flag

    def dfs(self, v_start, v_end=None) -> []:
        """
        Performs DFS and returns a list of vertices visited during the search, in the order they were visited. Takes
        index of the start vertex and optional end vertex index.
        If v_start is not in the graph, returns []. If v_end is not in the graph, proceeds as if there was no end
        vertex. Picks vertex index in ascending order.
        """
        visited_v = []
        dfs_stack = deque()  # stack methods: append(), pop() - LIFO
        # base case: start not in graph or start and end are the same
        if v_start >= self.v_count:
            visited_v = []
        elif v_start == v_end:
            visited_v.append(v_start)

        else:
            # add start to the empty stack
            cur = v_start
            dfs_stack.append(cur)

            # traverse graph
            while cur != v_end and len(dfs_stack) > 0:
                cur = dfs_stack.pop()
                # if v is not in visited v
                if cur not in visited_v:
                    # add v to the set
                    visited_v.append(cur)
                    # push direct successors to stack
                    row = self.adj_matrix[cur]
                    # reverse order
                    for i in range(self.v_count-1, -1, -1):
                        if row[i] != 0:
                            dfs_stack.append(i)
        return visited_v

    def bfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during BFS search
        Vertices are picked in alphabetical order
        If the starting vertex is not in the list, return an empty list. If the name of the end vertex is not in the
        list, proceed as if there is no end vertex.
        """
        visited_v = []
        bfs_queue = deque()  # queue methods: append(), popleft() - FIFO
        # base case: start not in graph or start and end are the same
        if v_start >= self.v_count:
            visited_v = []
        elif v_start == v_end:
            visited_v.append(v_start)
        else:
            # add start to the empty queue
            cur = v_start
            bfs_queue.append(cur)

            while cur != v_end and len(bfs_queue) != 0:
                cur = bfs_queue.popleft()
                if cur not in visited_v:
                    visited_v.append(cur)
                    # for each direct successor-v', if v' is not in visited enqueue it
                    row = self.adj_matrix[cur]
                    for i in range(self.v_count):
                        if row[i] > 0 and i not in visited_v:
                            bfs_queue.append(i)
        return visited_v

    def has_cycle(self):
        """
        Returns True if there is at least one cycle in the graph. False if the graph is acyclic.
        """
        print(self.get_edges())
        # base case, not enough vertices to have cycle
        if self.v_count <= 2:
            return False
        else:
            visited = []
            for vertex in range(self.v_count):
                if self.has_cycle_helper(vertex):
                    return True
            return False

    def dijkstra(self, src: int) -> []:
        """
        TODO: Write this implementation
        """
        pass

    def get_connected_vertices(self, src: int) -> []:
        """
        Helper: Takes a given src vertex index and returns a list of destination vertex indices that share a directed
        edge with the source. Used with is_valid_path().
        """
        if 0 <= src < self.v_count:
            vertices = []
            row = self.adj_matrix[src]
            for i in range(self.v_count):
                if row[i] > 0:
                    vertices.append(i)
            return vertices

# -----------------HELPERS------------------------------------------
    def has_cycle_helper(self, v_start: str):
        """
        Helper for has_cycle. Modified DFS. Recursively goes through graph, looking for vertex v' with back edge to a
        previously visited node that is not the parent v (vertex we started at).
        """
        visited_v = []
        dfs_stack = deque()  # stack methods: append(), pop() - LIFO
        leaves = []
        # base case: start not in graph or start and end are the same
        if v_start >= self.v_count:
            visited_v = []

        else:
            # add start to the empty stack
            cur = v_start
            dfs_stack.append(cur)

            # traverse graph
            while len(dfs_stack) > 0:
                cur = dfs_stack.pop()
                # check if cur is a leaf
                if sum(self.adj_matrix[cur]) == 0:
                    leaves.append(cur)
                # if v is not in visited v

                if cur not in visited_v:
                    # add v to the set
                    visited_v.append(cur)
                    # push direct successors to stack
                    row = self.adj_matrix[cur]
                    # reverse order
                    for i in range(self.v_count - 1, -1, -1):
                        if row[i] != 0:
                            dfs_stack.append(i)

                # make sure cur has been visited before, is not a leaf, and there is an edge between prev and cur
                # vertices
                elif cur in visited_v and cur not in leaves and self.adj_matrix[visited_v[-1]][cur] > 0:
                    return True
        return False


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

    print("\nRandom - bfs() example 2")
    print("--------------------------------------")
    edges = [(7, 0, 11), (8, 9, 6), (2, 5, 17), (4, 10, 11), (6, 9, 15), (3, 10, 20), (4, 0, 8), (12, 0, 1), (10, 12, 4),
             (4, 7, 4), (0, 5, 6), (4, 1, 4), (10, 0, 16), (8, 1, 2)]
    g = DirectedGraph(edges)
    for start in range(4, 5):
        print(f'{start} BFS:{g.bfs(start)}')



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

    print("\nRandom - method has_cycle() example 2")
    print("----------------------------------")
    edges = [(0, 1, 15), (0, 8, 20), (1, 3, 5), (3, 7, 6), (3, 8, 5), (4, 3, 5), (4, 8, 12), (5, 2, 16), (6, 3, 3), (7, 5, 2),
             (8, 12, 16), (9, 5, 20), (11, 1, 12), (11, 8, 2)]
    g = DirectedGraph(edges)

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
