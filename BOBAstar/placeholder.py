### Cost_Bounded_forward_Astar
### Cost_type = 1 for fuel cost, 2 for risk cost
def Cost_Bounded_forward_Astar_f1f2(graph_object, start_state, goal_state, upper_bounds_global): 
    # Initialize the priority queue with the start state.
    open_list = []
    closed = []
    f1min = {node: np.inf for node in graph_object.visibility_graph.nodes()}    
    ### node is of the form (f1, g1, f2, g2, state)
    heapq.heappush(open_list, (0, 0, 0, 0, start_state))
    
    while open_list:
        current_f1, current_g1, current_f2, current_g2, current_state = heapq.heappop(open_list)
        
        if current_state not in closed:
            ### Check if the current state is the goal state
            if current_state == goal_state:
                update_global_upper_bound("ub2forward", upper_bounds_global, current_g2)
                update_heuristic(graph_object, current_state, current_g1, Type="backward", Cost_type="fuel_cost")
                f1min[current_state] = current_g1
                closed.append(current_state)
                print(f"The ub2forward is updated to {upper_bounds_global['ub2forward']}, and goal_g1 is {current_g1}")
                print(f"fuel cost to goal: {current_g1}, risk cost to goal: {current_g2}")
                continue
            
            ### Check whether the current state exceeds the global upper bound
            with ub_lock:
                ub_g1_forward = upper_bounds_global["ub1forward"]
                ub_g2_forward = upper_bounds_global["ub2forward"]
            
            if current_g1 > ub_g1_forward or current_g2 > ub_g2_forward:
                ### Bascially path from this node is not feasible
                update_heuristic(graph_object, current_state, np.inf, Type="backward", Cost_type="fuel_cost")
                closed.append(current_state)
                continue
            
            ### Update the heuristic
            update_heuristic(graph_object, current_state, current_g1, Type="backward", Cost_type="fuel_cost")
            ### As this a new state not in closed list, this is the min cost1 for this state, thus we can upper bound the cost2 for this state 
            update_upper_bound(graph_object, current_state, current_g2, Type="forward", Cost_type="risk_cost")
            
            ### Expand successors
            for successor in graph_object.visibility_graph.successors(current_state):
                if successor not in closed:
                    with ub_lock:
                        edge_data = graph_object.visibility_graph.edges[current_state, successor]
                        node_data = graph_object.visibility_graph.nodes[successor]
                    
                    successor_g1 = current_g1 + edge_data["fuel_cost"]
                    successor_g2 = current_g2 + edge_data["risk_cost"]
                    
                    successor_f1 = successor_g1 + node_data["heuristic_cost"]["forward"]["fuel_cost"]
                    successor_f2 = successor_g2 + node_data["heuristic_cost"]["forward"]["risk_cost"]
                    
                    ### If the f1 and f2 values of the successor exceed the global upper bound, skip it
                    if successor_f1 > ub_g1_forward or successor_f2 > ub_g2_forward:
                        update_heuristic(graph_object, current_state, np.inf, Type="backward", Cost_type="fuel_cost")
                        closed.append(successor)
                        continue
                    #
                    ### The successor is in open list, but its value less than the current g1 value, update the g1min value and put it in the open list
                    if successor_f1 < f1min[successor]:
                        f1min[successor] = successor_f1
                        heapq.heappush(open_list, (successor_f1, successor_g1, successor_f2, successor_g2, successor))
                    
            closed.append(current_state)

### Cost_Bounded_forward_Astar
### Cost_type = 1 for fuel cost, 2 for risk cost
def Cost_Bounded_backward_Astar_f2f1(graph_object, start_state, goal_state, upper_bounds_global): 
    # Initialize the priority queue with the start state.
    open_list = []
    closed = []
    f2min = {node: np.inf for node in graph_object.visibility_graph.nodes()}    
    ### node is of the form (f1, g1, f2, g2, state)
    heapq.heappush(open_list, (0, 0, 0, 0, goal_state))
    
    while open_list:
        current_f2, current_g2, current_f1, current_g1, current_state = heapq.heappop(open_list)
        
        if current_state not in closed:
            ### Check if the current state is the goal state
            if current_state == start_state:
                update_global_upper_bound("ub1backward", upper_bounds_global, current_g1)
                update_heuristic(graph_object, current_state, current_g2, Type="forward", Cost_type="risk_cost")
                f2min[current_state] = current_g2
                closed.append(current_state)
                print(f"The ub1backward is updated to {upper_bounds_global['ub1backward']}, and goal_g2 is {current_g2}")
                continue
            
            with ub_lock:
                ub_f1_backward = upper_bounds_global["ub1backward"]
                ub_f2_backward = upper_bounds_global["ub2backward"]
            
            ### Check whether the current state exceeds the global upper bound
            if current_f1 > ub_f1_backward or current_f2 > ub_f2_backward:
                ### Bascially path from this node is not feasible
                update_heuristic(graph_object, current_state, np.inf, Type="forward", Cost_type="risk_cost")
                closed.append(current_state)
                continue
            
            update_heuristic(graph_object, current_state, current_g2, Type="forward", Cost_type="risk_cost")
            ### As this a new state not in closed list, this is the min cost2 for this state, thus we can upper bound the cost1 for this state 
            update_upper_bound(graph_object, current_state, current_g1, Type="backward", Cost_type="fuel_cost")
            
            # ### Check whether the current state exceeds the local upper bound
            # if current_g1 > graph_object.visibility_graph[current_state]["upper_bound"]["forward"] or current_g2 > graph_object.visibility_graph[current_state]["upper_bound"]["forward"]:
            #     ### Path possible from this this node, but the current path won't be parato optimal
            #     continue
            
            ### Expand successors
            for successor in graph_object.visibility_graph.successors(current_state):
                if successor not in closed:
                    with ub_lock:
                        edge_data = graph_object.visibility_graph.edges[current_state, successor]
                        node_data = graph_object.visibility_graph.nodes[successor]
                    
                    successor_g1 = current_g1 + edge_data["fuel_cost"]
                    successor_g2 = current_g2 + edge_data["risk_cost"]
                    
                    successor_f1 = successor_g1 + node_data["heuristic_cost"]["backward"]["fuel_cost"]
                    successor_f2 = successor_g2 + node_data["heuristic_cost"]["backward"]["risk_cost"]
                    
                    ### If the f1 and f2 values of the successor exceed the global upper bound, skip it
                    if successor_f1 > ub_f1_backward or successor_f2 > ub_f2_backward:
                        update_heuristic(graph_object, current_state, np.inf, Type="forward", Cost_type="risk_cost")
                        closed.append(successor)
                        continue
                    
                    ### The successor is in open list, but its value less than the current g1 value, update the g1min value and put it in the open list
                    if successor_f2 < f2min[successor]:
                        f2min[successor] = successor_f2
                        heapq.heappush(open_list, ( successor_f2, successor_g2, successor_f1, successor_g1, successor))
                    
            closed.append(current_state)

## Cost_Bounded_forward_Astar
## Cost_type = 1 for fuel cost, 2 for risk cost
def Cost_Bounded_forward_Astar_f2f1(graph_object, start_state, goal_state, upper_bounds_global): 
    # Initialize the priority queue with the start state.
    open_list = []
    closed = []
    f2min = {node: np.inf for node in graph_object.visibility_graph.nodes()}    
    ### node is of the form (f2, g2, f1, g1, state)
    heapq.heappush(open_list, (0, 0, 0, 0, start_state))
    
    while open_list:
        current_f2, current_g2, current_f1, current_g1, current_state = heapq.heappop(open_list)
        
        if current_state not in closed:
            ### Check if the current state is the goal state
            if current_state == goal_state:
                update_global_upper_bound("ub1forward", upper_bounds_global, current_g1)
                print(f"The current g1 is {current_g1} and current g2 is {current_g2}")
                update_heuristic(graph_object, current_state, current_g2, Type="backward", Cost_type="risk_cost")
                f2min[current_state] = current_g2
                closed.append(current_state) 
                continue
            
            ### Check whether the current state exceeds the global upper bound
            with ub_lock:
                ub_g1_forward = upper_bounds_global["ub1forward"]
                ub_g2_forward = upper_bounds_global["ub2forward"]
            
            if current_g1 > ub_g1_forward or current_g2 > ub_g2_forward:
                ### Bascially path from this node is not feasible
                update_heuristic(graph_object, current_state, np.inf, Type="backward", Cost_type="risk_cost")
                closed.append(current_state)
                continue
            
            update_heuristic(graph_object, current_state, current_g2, Type="backward", Cost_type="risk_cost")
            ### As this a new state not in closed list, this is the min cost2 for this state, thus we can upper bound the cost1 for this state 
            update_upper_bound(graph_object, current_state, current_g1, Type="forward", Cost_type="fuel_cost")
            
            # ### Check whether the current state exceeds the local upper bound
            # if current_g1 > graph_object.visibility_graph[current_state]["upper_bound"]["forward"] or current_g2 > graph_object.visibility_graph[current_state]["upper_bound"]["forward"]:
            #     ### Path possible from this this node, but the current path won't be parato optimal
            #     continue
            
            ### Expand successors
            for successor in graph_object.visibility_graph.successors(current_state):
                if successor not in closed:
                    with ub_lock:
                        edge_data = graph_object.visibility_graph.edges[current_state, successor]
                        node_data = graph_object.visibility_graph.nodes[successor]
                    
                    successor_g1 = current_g1 + edge_data["fuel_cost"]
                    successor_g2 = current_g2 + edge_data["risk_cost"]
                    
                    successor_f1 = successor_g1 + node_data["heuristic_cost"]["forward"]["fuel_cost"]
                    successor_f2 = successor_g2 + node_data["heuristic_cost"]["forward"]["risk_cost"]
                    
                    ### If the f1 and f2 values of the successor exceed the global upper bound, skip it
                    if successor_f2 > ub_g2_forward or successor_f2 > ub_g2_forward:
                        update_heuristic(graph_object, current_state, np.inf, Type="backward", Cost_type="risk_cost")
                        closed.append(successor)
                        continue
                    
                    ### The successor is in open list, but its value less than the current g1 value, update the g1min value and put it in the open list
                    if successor_f2 < f2min[successor]:
                        f2min[successor] = successor_f2
                        heapq.heappush(open_list, (successor_f2, successor_g2, successor_f1, successor_g1, successor))

            closed.append(current_state)

## Cost_Bounded_forward_Astar
## Cost_type = 1 for fuel cost, 2 for risk cost

def Cost_Bounded_backward_Astar_f1f2(graph_object, start_state, goal_state, upper_bounds_global): 
    # Initialize the priority queue with the start state.
    open_list = []
    closed = []
    f1min = {node: np.inf for node in graph_object.visibility_graph.nodes()}    
    ### node is of the form (f1, g1, f2, g2, state)
    heapq.heappush(open_list, (0, 0, 0, 0, goal_state))
    
    while open_list:
        current_f1, current_g1, current_f2, current_g2, current_state = heapq.heappop(open_list)
        
        if current_state not in closed:
            ### Check if the current state is the goal state
            if current_state == start_state:
                update_global_upper_bound("ub2backward", upper_bounds_global, current_g2)
                update_heuristic(graph_object, current_state, current_g1, Type="forward", Cost_type="fuel_cost")
                print(f"The current g1 is {current_g1} and current g2 is {current_g2}")
                f1min[current_state] = current_g1
                closed.append(current_state) 
                continue
            
            with ub_lock:
                ub_f1_backward = upper_bounds_global["ub1backward"]
                ub_f2_backward = upper_bounds_global["ub2backward"]
            
            ### Check whether the current state exceeds the global upper bound
            if current_g1 > ub_f1_backward or current_g2 > ub_f2_backward:
                ### Bascially path from this node is not feasible
                update_heuristic(graph_object, current_state, np.inf, Type="forward", Cost_type="fuel_cost")
                closed.append(current_state)
                continue
            
            update_heuristic(graph_object, current_state, current_g1, Type="forward", Cost_type="fuel_cost")
            ### As this a new state not in closed list, this is the min cost2 for this state, thus we can upper bound the cost1 for this state 
            update_upper_bound(graph_object, current_state, current_g2, Type="backward", Cost_type="risk_cost")
            
            ### Expand successors
            for successor in graph_object.visibility_graph.successors(current_state):
                if successor not in closed:
                    with ub_lock:
                        edge_data = graph_object.visibility_graph.edges[current_state, successor]
                        node_data = graph_object.visibility_graph.nodes[successor]
                    
                    successor_g1 = current_g1 + edge_data["fuel_cost"]
                    successor_g2 = current_g2 + edge_data["risk_cost"]
                    
                    successor_f1 = successor_g1 + node_data["heuristic_cost"]["backward"]["fuel_cost"]
                    successor_f2 = successor_g2 + node_data["heuristic_cost"]["backward"]["risk_cost"]
                    
                    ### If the f1 and f2 values of the successor exceed the global upper bound, skip it
                    if successor_f1 > ub_f1_backward or successor_f2 > ub_f2_backward:
                        update_heuristic(graph_object, current_state, np.inf, Type="forward", Cost_type="fuel_cost")
                        closed.append(successor)
                        continue
                    
                    ### The successor is in open list, but its value less than the current g1 value, update the g1min value and put it in the open list
                    if successor_f1 < f1min[successor]:
                        f1min[successor] = successor_f1
                        heapq.heappush(open_list, ( successor_f1, successor_g1, successor_f2, successor_g2, successor))
                    
            closed.append(current_state)
            
            

def biobjective_search_forward( graph_object, start_state="s", goal_state="g", reduce_factor=1):
    """
    Performs a biobjective search (fuel cost and risk cost) over the visibility graph.
    
    Args:
        graph_object: The graph object containing the visibility graph, and other relevant information.
        start_state: Identifier for the start node (default "s").
        goal_state: Identifier for the goal node (default "g").
        reduce_factor: A factor used in pruning dominated paths.
        
    Returns:
        sols: A dict mapping each state to a list of solution tuples.
        g2_min: A dict mapping each state to its minimum risk cost.
    """
    visibility_graph = graph_object.visibility_graph
    max_risk_limit, acceptable_risk_limit = graph_object.max_risk_limit, graph_object.acceptable_risk_limit

    # print("All states:", all_states)
    open_set = []
    sols = {state: [] for state in all_states}

    # The start node is represented as a tuple: (f1, f2, g1, g2, state)
    # and its parent is set to None.
    start_node = (0, 0, 0, 0, start_state)
    heapq.heappush(open_set, [start_node, (None, None, None, None, None)])

    while open_set:
        current_node, parent_node = heapq.heappop(open_set)
        current_f1, current_f2, current_g1, current_g2, current_state = current_node
        # print(f"current_state: {current_state}, current_f1: {current_f1}, current_f2: {current_f2}, current_g1: {current_g1}, current_g2: {current_g2}")  
        ### Check the global upper bound
        if current_f1 > g1_min["s"]:
            # print("global upper bound reached for g1")
            break
        
        # Prune if the current risk cost is dominated
        if (current_g2 >= reduce_factor*g2_min[current_state] or
            current_f2 >= reduce_factor*g2_min["g"]):
            continue
        
        ### Update the heuristic if the current state is reached for the first time
        if g2_min[current_state] == np.inf:
            update_heuristic(graph_object, current_state, current_g1, Type="backward", Cost_type="fuel_cost")
        
        update_g_min(g1_min, g2_min, current_state, current_g2, type=2)
        sols[current_state].append([current_node, parent_node])
        
        # Stop expanding if the goal is reached
        if current_state == goal_state:
            # print("Goal state reached")
            continue
        
        # Expand successors of the current state
        for successor in visibility_graph.successors(current_state):
            edge_data = visibility_graph.edges[current_state, successor]
            g1 = current_g1 + edge_data['fuel_cost']
            f1 = g1 + visibility_graph.nodes[successor]['heuristic_cost']["forward"]["fuel_cost"]
            g2 = current_g2 + edge_data['risk_cost']
            # For risk, we use a zero heuristic.
            f2 = g2 + visibility_graph.nodes[successor]['heuristic_cost']["forward"]["risk_cost"]

            # print(f"successor: {successor}, g1: {g1}, f1: {f1}, g2: {g2}, f2: {f2}")
            # Prune dominated successors
            with ub_lock:
                if (g2 >= reduce_factor * g2_min[successor] or
                    f2 >= reduce_factor * g2_min["g"]):
                    continue

            with ub_lock:
                if f1 >= g1_min[successor]:
                    # print("f1 >= g1_min[successor]")
                    continue
            
            child_node = (f1, f2, g1, g2, successor)
            heapq.heappush(open_set, [child_node, current_node])
            
    return sols, g2_min