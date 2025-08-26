# debug_graph_joins.py
import pickle
import networkx as nx

def debug_specific_join():
    """Debug the specific join between tblOApplicationMaster and tblIndustry"""
    
    with open("E:/Github/sql-ai-agent/data/join_graph/join_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    
    print("DEBUGGING JOIN BETWEEN tblOApplicationMaster AND tblIndustry")
    print("=" * 60)
    
    # Check direct connection
    if G.has_edge('tblOApplicationMaster', 'tblIndustry'):
        edge_data = G.get_edge_data('tblOApplicationMaster', 'tblIndustry')
        print("DIRECT EDGE: tblOApplicationMaster -> tblIndustry")
        for key, value in edge_data.items():
            print(f"  {key}: {value}")
    
    if G.has_edge('tblIndustry', 'tblOApplicationMaster'):
        edge_data = G.get_edge_data('tblIndustry', 'tblOApplicationMaster')
        print("\nDIRECT EDGE: tblIndustry -> tblOApplicationMaster")
        for key, value in edge_data.items():
            print(f"  {key}: {value}")
    
    # Check path through tblCounterparty
    print(f"\nPATH ANALYSIS:")
    try:
        path = nx.shortest_path(G, 'tblOApplicationMaster', 'tblIndustry')
        print(f"Shortest path: {' -> '.join(path)}")
        
        # Show each edge in the path
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            edge_data = G.get_edge_data(source, target, {})
            
            print(f"\nEDGE {i+1}: {source} -> {target}")
            for key, value in edge_data.items():
                print(f"  {key}: {value}")
                
    except nx.NetworkXNoPath:
        print("No path found")
    
    # Check all edges involving these tables
    print(f"\nALL tblOApplicationMaster CONNECTIONS:")
    for neighbor in G.neighbors('tblOApplicationMaster'):
        edge_data = G.get_edge_data('tblOApplicationMaster', neighbor, {})
        source_col = edge_data.get('source_column', 'unknown')
        target_col = edge_data.get('target_column', 'unknown')
        print(f"  -> {neighbor}: {source_col} -> {target_col}")
    
    print(f"\nALL tblIndustry CONNECTIONS:")
    for neighbor in G.neighbors('tblIndustry'):
        edge_data = G.get_edge_data('tblIndustry', neighbor, {})
        source_col = edge_data.get('source_column', 'unknown')
        target_col = edge_data.get('target_column', 'unknown')
        print(f"  -> {neighbor}: {source_col} -> {target_col}")

# Run the debug
debug_specific_join()
