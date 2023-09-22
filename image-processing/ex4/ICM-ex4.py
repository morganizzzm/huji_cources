import networkx as nx
import metis




def create_dict(n):
    s = dict()
    for i in range(n):
        for j in range(n):
            s[(i,j)] = 0
    return s


def add_edges(s, n):
    for i in range(n):
        for j in range(n):
            if i<n-1 and j<n-1:
                s[(i+1, j)] +=1
                s[(i, j)] += 1
                s[(i, j+1)] += 1
                s[(i, j)] += 1

            if i == n-1 and j!= n-1:
                s[(i, j+1)] +=1
                s[(i, j)] += 1

            if i!=n-1 and j == n-1:
                s[(i+1, j)] +=1
                s[(i, j)] += 1
    print(s)
    return s


def compute_max_neighbours(n):
    s = create_dict(n)
    edges = add_edges(s, n)
    max_value = max(s.values())
    print(max_value)
    for key in edges.keys():
        if (edges[key] == max_value):
            print(key)







if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (3, 4)])

    # Add node weights to graph
    for i, value in enumerate([1, 3, 2, 4, 3]):
        G.node[i]['node_value'] = value

    # tell METIS which node attribute to use for
    G.graph['node_weight_attr'] = 'node_value'

    # Get at MOST two partitions from METIS
    (cut, parts) = metis.part_graph(G, 2)
    # parts == [0, 0, 0, 1, 1]

    # Assuming you have PyDot installed, produce a DOT description of the graph:
    colors = ['red', 'blue']
    for i, part in enumerate(parts):
        G.node[i]['color'] = colors[part]
    nx.nx_pydot.write_dot(G, 'example.dot')