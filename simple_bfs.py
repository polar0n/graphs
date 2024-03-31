from graph_primitives import Vertex, DirectedEdge as Edge, DirectedGraph as Graph


def dynastatic_vertices(labels:list[str]):
    '''Dynamically create static variables into the __main__ top-level environment.'''
    for label in labels:
        globals()[label] = Vertex(label)


dynastatic_vertices([chr(i) for i in range(97, 123)])

# vertices = [Vertex(i) for i in range(1, 12)]

# edges = [
#     Edge(vertices[0], vertices[2]),
#     Edge(vertices[2], vertices[4]),
#     Edge(vertices[4], vertices[0]),
#     Edge(vertices[2], vertices[10]),
#     Edge(vertices[4], vertices[8]),
#     Edge(vertices[4], vertices[6]),
#     Edge(vertices[10], vertices[5]),
#     Edge(vertices[10], vertices[7]),
#     Edge(vertices[8], vertices[7]),
#     Edge(vertices[7], vertices[5]),
#     Edge(vertices[5], vertices[9]),
#     Edge(vertices[9], vertices[7]),
#     Edge(vertices[1], vertices[9]),
#     Edge(vertices[8], vertices[3]),
#     Edge(vertices[8], vertices[1]),
#     Edge(vertices[1], vertices[3]),
#     Edge(vertices[3], vertices[6]),
#     Edge(vertices[6], vertices[8])
# ]

vertices = [
    s, v, w, t
]

edges = [
    Edge(s, v, 1),
    Edge(s, w, 4),
    Edge(v, w, 2),
    Edge(v, t, 6), 
    Edge(w, t, 3)
]

G = Graph(vertices, edges)

print(G.dijkstra_heap(s, t))
