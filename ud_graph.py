# Course: CS 261
# Author: Jada Young
# Assignment: Asignment 6 - Undirected Graph
# Description: Implementation of Undirected Graph via Adjacency List. Supports the following types of graph:
# undirected, unweighted, no duplicate edges, no loops. Cycles are allowed.
# Included methods: 1) add_vertex(), add_edge(); 2) remove_edge(), remove_vertex; 3) get_vertices(), get_edges(); 4)
# is_valid_path(), dfs(), bfs(); 5) count_connected_components(), has_cycle()

import heapq
import queue
from collections import deque


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Add new vertex to the graph as key with value set to empty list. Does nothing if vertex is already present.
        """
        # if adjacency list is empty or key not present
        if len(self.adj_list) == 0 or v not in self.adj_list:
            self.adj_list[v] = []

    def add_edge(self, u: str, v: str) -> None:
        """
        Add edge to the graph by connecting two vertices with provided name. If either (or both) vertex names do not
        exist, the method first creates them. If an edge already exists in the graph, or if u and v are the same,
        the method does nothing.
        """
        # Same vertex check
        if u != v:
            # Check if vertices are in graph
            if u not in self.adj_list:
                self.add_vertex(u)
            if v not in self.adj_list:
                self.add_vertex(v)
            # Check if edge already exists, add if not
            if v not in self.adj_list[u] and u not in self.adj_list[v]:
                self.adj_list[u].append(v)
                self.adj_list[v].append(u)

    def remove_edge(self, v: str, u: str) -> None:
        """
        Remove edge from the graph. If either or both vertex names do not exist, or there is not an edge between
        them, does nothing.
        """
        if v in self.adj_list and u in self.adj_list:
            # check for edge
            if v in self.adj_list[u] and u in self.adj_list[v]:
                self.adj_list[u].remove(v)
                self.adj_list[v].remove(u)

    def remove_vertex(self, v: str) -> None:
        """
        Remove vertex and all connected edges. If the vertex does not exist, does nothing.
        """
        if v in self.adj_list:
            # go through v's edges and remove
            # using copy
            v_edges = []
            for vertex in self.adj_list[v]:
                v_edges.append(vertex)
            for vertex in v_edges:
                self.remove_edge(v, vertex)
            # delete v
            del self.adj_list[v]

    def get_vertices(self) -> []:
        """
        Return list of vertices in the graph (any order)
        """
        return list(self.adj_list.keys())

    def get_edges(self) -> []:
        """
        Return list of edges in the graph (any order). Each edge is returned as a tuple of two incident vertex names
        """
        edge_list = []
        for vertex in self.adj_list:
            for adj in self.adj_list[vertex]:
                if (adj, vertex) not in edge_list:
                    group = vertex, adj
                    edge_list.append(group)
        return edge_list

    def is_valid_path(self, path: []) -> bool:
        """
        Return true if provided path is valid, False otherwise. Empty paths are valid. Implemented with is_reachable().
        """
        valid_flag = True
        index = 0

        # base case, single ir empty path given
        if len(path) == 0:
            pass
        elif len(path) == 1:
            valid_flag = path[0] in self.adj_list
        else:
            while valid_flag and index < len(path)-1:  # second to last vertex in path
                valid_flag = self.is_reachable(path[index], path[index+1])
                index += 1

        return valid_flag

    def dfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during DFS search
        Vertices are picked in alphabetical order
        If the starting vertex is not in the list, return an empty list. If the name of the end vertex is not in the
        list, proceed as if there is no end vertex.
        """
        visited_v = []
        dfs_stack = deque()  # stack methods: append(), pop() - LIFO
        # base case: start not in graph or start and end are the same
        if v_start not in self.adj_list:
            visited_v = []
        elif v_start == v_end:
            visited_v.append(v_start)

        else:
            # add start to the empty stack
            cur = v_start
            dfs_stack.append(cur)

            while cur != v_end and len(dfs_stack) != 0:
                # if stack is not empty, pop a vertex v
                cur = dfs_stack.pop()
                # if v is not in visited v
                if cur not in visited_v:
                    # add v to the set
                    visited_v.append(cur)
                    # add each vertex that is a direct successor of v to the stack
                    # since in lexo order, add last vertex first
                    rev_edges = sorted(self.adj_list[cur], reverse=True)
                    for edge in rev_edges:
                        dfs_stack.append(edge)
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
        if v_start not in self.adj_list:
            visited_v = []
        elif v_start == v_end:
            visited_v.append(v_start)
        else:
            # add start to the empty queue
            cur = v_start
            bfs_queue.append(cur)

            while cur != v_end and len(bfs_queue) != 0:
                cur = bfs_queue.popleft()
                ordered_edges = sorted(self.adj_list[cur])
                if cur not in visited_v:
                    visited_v.append(cur)
                    # for each direct successor-v', if v' is not in visited enqueue it
                    for edge in ordered_edges:
                        if edge not in visited_v:
                            bfs_queue.append(edge)
        return visited_v

    def count_connected_components(self):
        """
        Return number of connected components in the graph
        """
        # https://web.stanford.edu/class/archive/cs/cs161/cs161.1182/Lectures/Lecture10/CS161Lecture10.pdf
        # BFS/DFS Proof
        visited_vertices = set()
        components_count = 0
        vertices = self.get_vertices()
        # Each vertex that can be reached from a given vertex composes a single connected component
        # Go through every vertex, finding connected components one at a time
        for v in vertices:
            # Do not need to search from v if it is already part of a previous connected component
            if v not in visited_vertices:
                visited_vertices.add(v)
                con_components = self.bfs(v)
                components_count += 1
                for vertex in con_components:
                    visited_vertices.add(vertex)
        return components_count


    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """

# -------------------------HELPERS---------------------------------------------------------------
    def is_reachable(self, u: str, v: str) -> bool:
        """
        Helper: determines if a node is reachable from another node. Returns True if an edge exists between
        vertices, False otherwise.
        Used with: is_valid_path()
        """
        return v in self.adj_list[u] and u in self.adj_list[v]


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)

    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)

    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')

    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
