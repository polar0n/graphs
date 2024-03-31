from datastruct import Queue, Stack, MinHeap as Heap
from itertools import permutations, product
from singletons import Inf


INF = Inf()


class MissingVertexException(Exception):
    '''Raised when a graph is missing a vertex which is part of an edge.'''
    pass


class VertexNotInGraph(Exception):
    '''Raised when trying to access a vertex that is not part of a graph.'''
    pass


class Vertex:
    '''
    Vertex class implementation that stores all the created `Vertex` instances.
    When initializing a vertex with the same label as another previously initialized
    vertex, the constructor will return the instance to the previous `Vertex` object.
    ```
    >>> A1 = Vertex('A')
    >>> A2 = Vertex('A')
    >>> A1 is A2
    True
    ```
    '''
    _instances = dict()


    def __new__(cls, *args, **kwargs):
        # if cls._instances.get(args[0])
        if len(args) or len(kwargs):
            label = list(args)
            label.append(kwargs.get('label'))
            label = label[0]
            returning_instance = cls._instances.get(label)
            if returning_instance:
                return returning_instance
        else:
            return super().__new__(cls)
        new_instance = super().__new__(cls)
        cls._instances[label] = new_instance
        return new_instance


    def __init__(self, label:str=None):
        self.label = label


    @staticmethod
    def generate_vertices(labels: list[str]):
        '''
        Generate static class attributes with the list of labels that can be accessed statically.
        Ex. `Vertex.generate_labels(['a', 'b'])\nVertex.a -> Vertex(a)`
        '''
        for label in labels:
            setattr(Vertex, label, Vertex(label))


    def __repr__(self) -> str:
        return f'Vertex({self.label})'


    def __str__(self) -> str:
        return self.label


    def __eq__(self, other) -> bool:
        return other and other.label and self.label == other.label
    

    def __ne__(self, other) -> bool:
        return self.__eq__(other)
    

    def __hash__(self) -> int:
        return hash(repr(self.label))


class UndirectedEdge:
    '''
    Class implementation of an undirected edge `(v, w)` with one endpoint `v` and other `w`.
    The order of the endpoints is permutable
    ```
    >>> UndirectedEdge(v, w) == UndirectedEdge(w, v)
    True
    ```
    '''
    def __init__(self, endpoint1:Vertex, endpoint2:Vertex):
        self.e1 = endpoint1
        self.e2 = endpoint2
        self.endpoints = {endpoint1, endpoint2}


    def __repr__(self) -> str:
        return f'Edge({self.e1}, {self.e2})'


    def __str__(self) -> str:
        return f'({self.e1}, {self.e2})'


    def __eq__(self, other) -> bool:
        return other and not (other.endpoints - self.endpoints)


    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


    def __hash__(self) -> int:
        return hash((self.e1, self.e2))


    def get_other_endpoint(self, vertex:Vertex) -> Vertex:
        '''
        Returns the other endpoint when one is given.
        For edge `(v, w)`, a call to `.get_other_endpoint(v)` will return `w`.
        '''
        if vertex == self.e1:
            return self.e2
        return self.e1


class DirectedEdge:
    '''Class implementation of a directed edge `(v, w)` with a tail `v` and head `w`.'''
    def __init__(self, tail:Vertex, head:Vertex, length:int=1):
        self.tail = tail
        self.head = head
        self.length = length


    def __repr__(self) -> str:
        return f'Edge({self.tail}, {self.head})'


    def __str__(self) -> str:
        return f'({self.tail}, {self.head})'


    def __eq__(self, other) -> bool:
        return other and self.tail == other.tail and self.head == other.head


    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


    def __hash__(self) -> int:
        return hash(repr(self))


class UndirectedGraph:
    '''
    An object-oriented approach to an undirected graph which represents edges and vertices within the graph
    as two lists, one of vertices and one of edges. Does not use adjacency list representation.
    '''
    def __init__(self, V:list[Vertex], E:list[UndirectedEdge]):
        '''
        Initialize a new graph `G = (V, E)`, where `G` is the `UndirectedGraph` object,
        `V` is a list of `Vertex` objects, and `E` is a list of `UndirectedEdges` objects.
        '''
        self.vertices = V.copy()
        self.edges = E.copy()
        self._sort_edges()
        self._base_exploration_map = {vertex: False for vertex in self.vertices}


    def _sort_edges(self):
        '''Sort the edges into adjacency list representation for faster post-retrieval.'''
        # Verify if any edges have any unknown endpoints
        edge_vertices = set()
        for edge in self.edges:
            edge_vertices.add(edge.e1)
            edge_vertices.add(edge.e2)
        if edge_vertices.difference(self.vertices):
            raise MissingVertexException('One or more parsed vertices in edges are not part of vertex list.')

        self._sorted_edges = {vertex: list() for vertex in self.vertices}
        for edge in self.edges:
            self._sorted_edges.get(edge.e1).append(edge)
            self._sorted_edges.get(edge.e2).append(edge)

    def get_edges(self, vertex:Vertex) -> list[UndirectedEdge]:
        '''Return a list with all the edges pertaining to `vertex`.'''
        if vertex not in self.vertices:
            raise VertexNotInGraph(f'Vertex {vertex} not part of graph {self}')
        return self._sorted_edges[vertex]
    

    def bfs(self, s:Vertex, v:Vertex) -> bool:
        '''
        Breadth-first search (BFS) algorithm returns true if vertex `v` is reachable from vertex `s`, false otherwise.
        BFS algorithm explores the graph in layers.
        '''
        exploration_map = self._base_exploration_map.copy()
        Q = Queue(s)
        exploration_map[s] = True
        while not Q.empty():
            current_vertex = Q.get()
            for edge in self.get_edges(current_vertex):
                endpoint = edge.get_other_endpoint(current_vertex)
                if not exploration_map[endpoint]:
                    exploration_map[endpoint] = True
                    Q.put(endpoint)

        return exploration_map[v]


    def shortest_paths(self, s:Vertex) -> dict[Vertex, int]:
        '''
        Computes the shortest path distances of each vertex in graph from vertex `s`.
        If the edge objects do not have the `length` attribute, each edge is assumed to have length `1`.
        Returns a dict with keys as vertices and values as path distances.
        If there is no path from `s` to any given vertex within the graph, the shortest path distance to that vertex
        will be `INF`, which is `+inf`.
        '''
        exploration_map = self._base_exploration_map.copy()
        distances = {vertex: INF for vertex in self.vertices}
        distances[s] = 0
        Q = Queue(s)
        exploration_map[s] = True
        while not Q.empty():
            current_vertex = Q.get()
            for edge in self.get_edges(current_vertex):
                endpoint = edge.get_other_endpoint(current_vertex)
                if not exploration_map[endpoint]:
                    exploration_map[endpoint] = True
                    distances[endpoint] = distances[current_vertex] + 1
                    Q.put(endpoint)

        return distances


    def connected_components(self) -> dict[Vertex, int]:
        '''
        Returns a dictionary where keys represent vertices and their values represent the number of
        the strongly connected component to which each vertex belongs. The first strongly component
        always has number `1`.
        '''
        exploration_map = self._base_exploration_map.copy()
        cc = {vertex: -1 for vertex in self.vertices}
        numCC = 0
        for vertex in self.vertices:
            if exploration_map[vertex]:
                continue
            numCC += 1
            # BFS
            Q = Queue(vertex)
            exploration_map[vertex] = True
            while not Q.empty():
                v = Q.get()
                cc[v] = numCC
                for edge in self.get_edges(v):
                    endpoint = edge.get_other_endpoint(v)
                    if not exploration_map[endpoint]:
                        exploration_map[endpoint] = True
                        Q.put(endpoint)

        return cc
    

    def dfs(self, s:Vertex, v:Vertex) -> bool:
        '''
        Depth-first search (DFS) algorithm returns true if vertex `v` is reachable from vertex `s`, false otherwise.
        DFS explores the graph aggressively and explores the first outgoing vertex of the current vertex with its loop.
        '''
        exploration_map = self._base_exploration_map.copy()
        S = Stack(s)
        while not S.empty():
            current_vertex = S.pop()
            if exploration_map[current_vertex]:
                continue
            exploration_map[current_vertex] = True
            for edge in self.get_edges(current_vertex):
                S.push(edge.get_other_endpoint(current_vertex))

        return exploration_map[v]


class DirectedGraph:
    '''
    An object-oriented approach to a directed graph which represents edges and vertices within the graph
    as two lists, one of vertices and one of edges. Does not use adjacency list representation.
    '''

    def __init__(self, V:list[Vertex], E:list[DirectedEdge]):
        '''
        Initialize a new directed graph `G = (V, E)`, where `G` is the `DirectedGraph` object,
        `V` is a list of `Vertex` objects, and `E` is a list of `DirectedEdge` objects.
        '''
        self.vertices = V.copy()
        self.VERTICES = len(V)
        self.edges = E.copy()
        self.EDGES = len(E)
        self._sort_edges()
        self._base_exploration_map = {vertex: False for vertex in self.vertices}


    def _sort_edges(self):
        '''Sort the edges for faster post-retrieval.'''
        # Verify if any edges have any unknown edges
        edge_vertices = set()
        for edge in self.edges:
            edge_vertices.add(edge.tail)
            edge_vertices.add(edge.head)
        if edge_vertices.difference(self.vertices):
            raise MissingVertexException('One or more parsed vertices in edges are not part of vertex list.')

        self._sorted_edges = {vertex: list() for vertex in self.vertices}
        for edge in self.edges:
            self._sorted_edges.get(edge.tail).append(edge)


    def _sort_edges_incoming(self) -> dict[Vertex, DirectedEdge]:
        '''Sort the edges into the incoming edges of a vertex.'''
        inverted_edges = {vertex: list() for vertex in self.vertices}
        for edge in self.edges:
            inverted_edges[edge.head].append(edge)
        return inverted_edges
    

    def reverse_graph(self):
        '''Returns a `DirectedGraph` instance with the same vertices and reversed edges.'''
        reversed_edges = list()
        for edge in self.edges:
            reversed_edges.append(DirectedEdge(edge.head, edge.tail))
        return DirectedGraph(self.vertices, reversed_edges)


    def get_edges(self, vertex:Vertex) -> list[DirectedEdge]:
        '''
        Return a list with the outgoing edges of `vertex`; i.e., return all the
        edges where `vertex` is the tail.
        '''
        if vertex not in self.vertices:
            raise VertexNotInGraph(f'Vertex {vertex} not part of graph {self}')
        return self._sorted_edges[vertex]


    def get_sinks(self) -> list[Vertex]:
        '''
        Return the list of vertices that have only incoming edges.
        '''
        if hasattr(self, '_sinks'):
            return self._sinks

        sinks = list()
        for vertex in self.vertices:
            edges = self.get_edges(vertex)
            if not edges:
                if vertex not in sinks:
                    sinks.append(vertex)
                continue
        self._sinks = sinks
        return self._sinks
    

    def get_sources(self) -> list[Vertex]:
        '''
        Return the list of vertices that have only outgoing edges.
        '''
        if hasattr(self, '_sources'):
            return self._sources

        incoming_edges = self._sort_edges_incoming()
        sources = list()
        for vertex in self.vertices:
            if (not incoming_edges[vertex]) and \
                bool(self.get_edges(vertex)) and \
                (vertex not in sources):
                sources.append(vertex)

        self._sources = sources
        return self._sources


    def shortest_paths(self, s:Vertex) -> dict[Vertex, int]:
        '''
        Computes the shortest path distances of each vertex in graph from vertex `s`.
        If the edge objects do not have the `length` attribute, each edge is assumed to have length `1`.
        Returns a dict with keys as vertices and values as path distances.
        If there is no path from `s` to any given vertex within the graph, the shortest path distance to that vertex
        will be `INF`, which is `+inf`.
        '''
        exploration_map = self._base_exploration_map.copy()
        distances = {vertex: INF for vertex in self.vertices}
        distances[s] = 0
        Q = Queue(s)
        exploration_map[s] = True
        while not Q.empty():
            current_vertex = Q.get()
            for edge in self.get_edges(current_vertex):
                endpoint = edge.head
                if not exploration_map[endpoint]:
                    exploration_map[endpoint] = True
                    distances[endpoint] = distances[current_vertex] + 1
                    Q.put(endpoint)

        return distances
    

    def bfs(self, s:Vertex, v:Vertex) -> bool:
        '''
        Breadth-first search (BFS) algorithm returns true if vertex `v` is reachable from vertex `s`, false otherwise.
        BFS algorithm explores the graph in layers.
        '''
        exploration_map = self._base_exploration_map.copy()
        Q = Queue(s)
        exploration_map[s] = True
        while not Q.empty():
            current_vertex = Q.get()
            for edge in self.get_edges(current_vertex):
                endpoint = edge.head
                if not exploration_map[endpoint]:
                    exploration_map[endpoint] = True
                    Q.put(endpoint)

        return exploration_map[v]
    

    def dfs(self, s:Vertex, v:Vertex) -> bool:
        '''
        Depth-first search (DFS) algorithm returns true if vertex `v` is reachable from vertex `s`, false otherwise.
        DFS explores the graph aggressively and explores the first outgoing vertex of the current vertex with its loop.
        '''
        exploration_map = self._base_exploration_map.copy()
        S = Stack(s)
        while not S.empty():
            current_vertex = S.pop()
            if exploration_map[current_vertex]:
                continue
            exploration_map[current_vertex] = True
            for edge in self.get_edges(current_vertex):
                S.push(edge.head)

        return exploration_map[v]
        

    def topo_sort_recursive(self) -> dict[Vertex: int]:
        '''Return one topological sorting of the graph using a recursive
        depth-first search. Halting is not guaranteed.'''
        global current_label
        exploration_map = self._base_exploration_map.copy()
        current_label = len(self.vertices)
        f_map = dict()

        def dfs_topo(vertex):
            global current_label
            exploration_map[vertex] = True
            for edge in self.get_edges(vertex):
                head = edge.head
                if exploration_map[head]:
                    continue
                dfs_topo(head)
            f_map[vertex] = current_label
            current_label -= 1

        for vertex in self.vertices:
            if exploration_map[vertex]:
                continue
            dfs_topo(vertex)

        return f_map


    # def topo_sort(self) -> dict[Vertex, int]:
    #     '''Return one topological sorting of the graph.'''
    #     exploration_map = self._base_exploration_map.copy()
    #     current_label = self.VERTICES
    #     f_map = dict()

    #     for vertex in self.vertices:
    #         if exploration_map[vertex]:
    #             continue
    #         S = Stack(vertex)
    #         while not S.empty():
    #             current_vertex = S.pop()
    #             if exploration_map[current_vertex]:
    #                 continue
    #             exploration_map[current_vertex] = True
    #             for edge in self.get_edges(current_vertex):
    #                 S.push(edge.head)

    #     return f_map


    def full_topo_sort(self) -> list[dict[Vertex, int]]:
        '''
        Returns a list of all possible topological sortings of the graph. May be computationally expensive.
        '''
        exploration_map = self._base_exploration_map.copy()
        permutation_matrix = dict()

        # Find all the branching vertices and compute the permutations of those branches
        for current_vertex in self.vertices:
            permutation_matrix[current_vertex] = list(permutations(self.get_edges(current_vertex)))

        # Compute all possible combinations of those branches
        edge_combination_space = list(product(*permutation_matrix.values()))
        # Create list of all those combinations with each combinations as a sorted edge dictionary
        edge_combinations = list()
        for combination in edge_combination_space:
            comb = dict()
            for edges in zip(self.vertices, combination):
                comb[edges[0]] = edges[1]
            edge_combinations.append(comb)

        # Create source vertices permutations
        source_vertex_permutations = list(permutations(self.get_sources()))
        # It does not make any sense to compute the permutations of all the vertices.
        # Create a list of of all vertex permutations with the sources at the head of the lists
        vertices_permutations = list()
        # Get all the vertices without source vertices
        nonsource_vertices = list(set(self.vertices).difference(self.get_sources()))
        for source_vertex_permutation in source_vertex_permutations:
            vertices_permutations.append(list(source_vertex_permutation) + nonsource_vertices)

        f_maps = list()
        # Perform recursive topo-sort with dfs on each edge combination
        for edge_combination, vertices_permutation in product(edge_combinations, vertices_permutations):
            global current_label
            exploration_map = self._base_exploration_map.copy()
            current_label = self.VERTICES
            f_map = dict()

            def dfs_topo(vertex):
                global current_label
                exploration_map[vertex] = True
                for edge in edge_combination[vertex]:
                    head = edge.head
                    if exploration_map[head]:
                        continue
                    dfs_topo(head)
                f_map[vertex] = current_label
                current_label -= 1

            for vertex in vertices_permutation:
                if exploration_map[vertex]:
                    continue
                dfs_topo(vertex)

            if f_map in f_maps:
                continue
            print(*list(f_map.keys())[::-1], sep=' -> ')
            f_maps.append(f_map)

        return f_maps
    

    def strongly_connected_components(self) -> dict[Vertex, int]:
        '''
        Returns a dictionary with keys as vertices and values as numbers each representing the number
        of the strongly connected component (SCC) to which the vertex belongs. The first SCC number is `1`.
        A strongly connected component is a maximal subset `S` of `V` in which any vertex `v` can be reached by any
        other vertex in `S`. This method makes use of the `Kosaraju algorithm`.
        '''
        reversed_graph = self.reverse_graph()

        f_map = reversed_graph.topo_sort_recursive()
        # Sort the vertices based on their topological number
        f_sorted = sorted(f_map.keys(), key=lambda vertex: f_map.get(vertex))

        global num_scc, exploration_map, scc_map
        num_scc = 0
        exploration_map = self._base_exploration_map.copy()
        scc_map = dict()


        def dfs_scc(vertex):
            exploration_map[vertex] = True
            scc_map[vertex] = num_scc
            for edge in self.get_edges(vertex):
                if exploration_map[edge.head]:
                    continue
                dfs_scc(edge.head)


        for vertex in f_sorted:
            if exploration_map[vertex]:
                continue
            num_scc += 1
            dfs_scc(vertex)


        return scc_map
    

    # def get_meta_graph(self):
    #     '''
    #     Returns the meta-graph of this graph as an instance of `DirectedGraph` based on the results of the
    #     `.strongly_connected_components()` method. For more information refer to `.scc()`.
    #     '''
    #     scc_map = self.strongly_connected_components()
    #     vertices = [Vertex(label) for label in list(set(scc_map.values()))]        


    def scc(self) -> dict[Vertex, int]:
        '''
        Returns a dictionary with keys as vertices and values as numbers each representing the number
        of the strongly connected component (SCC) to which the vertex belongs. The first SCC number is `1`.
        A strongly connected component is a maximal subset `S` of `V` in which any vertex `v` can be reached by any
        other vertex in `S`. This method makes use of the `Kosaraju algorithm`.
        '''
        return self.strongly_connected_components()
    

    def dijkstra(self, start:Vertex, end:Vertex) -> dict[Vertex, int]:
        '''
        Computes `Dijkstra` shortest-path where `start` and `end` represent the starting and ending vertices respectively.
        Returns a dictionary with vertices as keys and values as Dijkstra scores. For shortest-path function implementing
        Dijkstra use `.dijkstra_path`.
        '''

        explored = {start}
        frontier_edges = self.get_edges(start)

        shortest_lengths = {vertex: INF for vertex in self.vertices}
        shortest_lengths[start] = 0
        while frontier_edges:
            dijkstra_scores = [(edge, shortest_lengths[edge.tail] + edge.length) for edge in frontier_edges]
            # Find the edge with the smallest Dijkstra Score
            min_edge, min_score = min(dijkstra_scores, key=lambda pair: pair[1])      # (edge, dijkstra_score)
            
            explored.add(min_edge.head)

            shortest_lengths[min_edge.head] = min_score

            frontier_edges = list()
            for vertex in explored:
                for edge in self.get_edges(vertex):
                    if edge.head in explored:
                        continue
                    frontier_edges.append(edge)

        return shortest_lengths


    def dijkstra_path(self, start:Vertex, end:Vertex) -> (list[Vertex] | None):
        explored = {start}
        frontier_edges = self.get_edges(start)
        predecessors = dict()

        shortest_lengths = {vertex: INF for vertex in self.vertices}
        shortest_lengths[start] = 0
        while frontier_edges:
            dijkstra_scores = [(edge, shortest_lengths[edge.tail] + edge.length) for edge in frontier_edges]
            # Find the edge with the smallest Dijkstra Score
            min_edge, min_score = min(dijkstra_scores, key=lambda pair: pair[1])      # (edge, dijkstra_score)

            explored.add(min_edge.head)
            shortest_lengths[min_edge.head] = min_score

            predecessors[min_edge.tail] = min_edge.head

            frontier_edges = list()
            for vertex in explored:
                for edge in self.get_edges(vertex):
                    if edge.head in explored:
                        continue
                    frontier_edges.append(edge)

        if shortest_lengths[end] == INF:
            return

        successors = dict(zip(predecessors.values(), predecessors.keys()))
        path = [end]
        last_vertex = end
        while last_vertex is not start:
            last_vertex = successors[last_vertex]
            path.append(last_vertex)

        return path
    

    def dijkstra_heap(self, start:Vertex, end:Vertex) -> dict[Vertex, int]:
        '''
        Implements `Dijkstra` shortest-path algorithm using `MinHeap` data structure for enhanced speed. For other
        information refer to `.dijkstra`.
        '''

        # Add .key and .len attribute to all vertices
        for vertex in self.vertices:
            setattr(vertex, 'key', INF)
            setattr(vertex, 'len', INF)
        # Set the attribute of the start vertex to 0
        start.key = 0
        
        explored = set()
        heap = Heap('key', self.vertices)
        path = list()

        while not heap.empty():
            w = heap.extract_min()
            path.append(w)
            explored.add(w)
            w.len = w.key
            for edge in self.get_edges(w):
                y = edge.head
                heap.delete(y)
                lengths = (y.key, w.len + edge.length)
                bigger = int(lengths[0] > lengths[1])
                y.key = lengths[bigger]
                heap.insert(y)

        # Remove .key and .len attributes from the vertices
        for vertex in self.vertices:
            delattr(vertex, 'key')
            delattr(vertex, 'len')
        return path


if __name__ == '__main__':
    print(dir(__name__))
