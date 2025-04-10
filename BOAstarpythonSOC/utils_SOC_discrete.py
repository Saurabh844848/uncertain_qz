import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import networkx as nx
import heapq

class GraphConstructionDiscretization:
    """
    Constructs a graph based on circle discretization and edge feasibility.
    
    Attributes:
        map_qz (list of tuples): Each tuple (cx, cy, radius) represents a circle.
        start (tuple): (x, y) coordinates of the start point.
        goal (tuple): (x, y) coordinates of the goal point.
        q_min, q_max, q_act: Numeric parameters.
        alpha, beta: Parameters for feasibility computations.
        discretization_angle (float): Angle step (in degrees) for discretization.
    """
    def __init__(self, map_qz, start, goal, q_min, q_max, alpha, beta, discretization_angle, discretization_SOC, max_risk_limit, acceptable_risk_limit):
        self.map_qz = map_qz
        self.start = start
        self.goal = goal
        self.q_min = q_min
        self.q_max = q_max
        # self.q_act = q_act
        self.alpha = alpha
        self.beta = beta
        self.discretization_angle = discretization_angle
        self.discretization_SOC = discretization_SOC
        self.max_risk_limit = max_risk_limit
        self.acceptable_risk_limit = acceptable_risk_limit

    def create_nodes(self):
        """
        Discretizes the circles and creates nodes with start labeled as "s" and goal as "g".
        
        Returns:
            node_positions (np.ndarray): Each row is [x, y, circle_index, q_act].
            index_map (dict): Maps (x, y, circle_index, q_act) to a node key.
            reverse_index_map (dict): Reverse mapping from node key to (x, y, circle_index, q_act).
        """
        angles = np.deg2rad(np.arange(0, 360 + self.discretization_angle, self.discretization_angle))
        self.SOC = np.arange(self.q_min, self.q_max + self.discretization_SOC, self.discretization_SOC)
        self.num_SOC = len(self.SOC)
        num_points_per_circle = len(angles)*self.num_SOC
        total_nodes = len(self.map_qz) * num_points_per_circle + 2*self.num_SOC
        node_positions = np.zeros((total_nodes, 4))

        # Use custom keys for start and goal: "s" for start and "g" for goal.
        index_map = {
            (self.start[0], self.start[1], 0, SOC_charge): i for i, SOC_charge in enumerate(self.SOC)
                    }
        
        index_map.update({
            (self.goal[0], self.goal[1], len(self.map_qz) + 1, SOC_charge): i+self.num_SOC for i, SOC_charge in enumerate(self.SOC)
        })
        
        self.reverse_index_map = {value: key for key, value in index_map.items()}

        # Assign start and goal positions in the node array.
        node_positions[:self.num_SOC, 0] = self.start[0]
        node_positions[:self.num_SOC, 1] = self.start[1]
        node_positions[:self.num_SOC, 2] = 0
        node_positions[:self.num_SOC, 3] = self.SOC
        
        node_positions[self.num_SOC:self.num_SOC*2, 0] = self.goal[0]
        node_positions[self.num_SOC:self.num_SOC*2, 1] = self.goal[1]
        node_positions[self.num_SOC:self.num_SOC*2, 2] = len(self.map_qz) + 1
        node_positions[self.num_SOC:self.num_SOC*2, 3] = self.SOC
        node_index = 2*self.num_SOC  # start index for circle nodes
        
        for circle_index, (cx, cy, radius, radius_in, risk_limit, toggle) in enumerate(self.map_qz, start=1):
            x_vals = cx + radius * np.cos(angles)
            y_vals = cy + radius * np.sin(angles)
            x_vals = np.repeat(x_vals, self.num_SOC)
            y_vals = np.repeat(y_vals, self.num_SOC)
            SOC_reps = np.tile(self.SOC, len(angles))
            
            node_positions[node_index:node_index + num_points_per_circle, 0] = x_vals
            node_positions[node_index:node_index + num_points_per_circle, 1] = y_vals
            node_positions[node_index:node_index + num_points_per_circle, 2] = circle_index
            node_positions[node_index:node_index + num_points_per_circle, 3] = SOC_reps
            
            for j in range(num_points_per_circle):
                key = (x_vals[j], y_vals[j], circle_index, SOC_reps[j])
                index_map[key] = node_index + j
                self.reverse_index_map[node_index + j] = key
            ### Updating the node index
            node_index += num_points_per_circle
        
        ### Adding a dummy node for the goal to be used in the path planning
        index_map[(self.goal[0], self.goal[1], len(self.map_qz) + 1, self.q_min)] = node_index
        self.reverse_index_map[node_index] = (self.goal[0], self.goal[1], len(self.map_qz) + 1, self.q_min)
        self.node_positions = node_positions
        
        return node_positions, index_map, self.reverse_index_map

    @staticmethod
    def point_to_segment_distance(px, py, x1, y1, x2, y2):
        """
        Computes the minimum distance from point (px, py) to the line segment defined by (x1, y1) and (x2, y2).
        """
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D
        param = dot / len_sq if len_sq != 0 else -1

        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx, yy = x1 + param * C, y1 + param * D

        dx = px - xx
        dy = py - yy
        return np.sqrt(dx * dx + dy * dy)
    
    def line_segment_intersects_circle(self, circle, point1, point2):
        """
        Checks if the line segment (point1 to point2) intersects the given circle.
        
        Args:
            circle (tuple): (center_x, center_y, radius)
            point1 (tuple): (x1, y1)
            point2 (tuple): (x2, y2)
            
        Returns:
            bool: True if the segment intersects the circle, False otherwise.
        """
        center_x, center_y, radius, radius_in, risk_limit, toggle = circle
        x1, y1 = point1
        x2, y2 = point2
        return self.point_to_segment_distance(center_x, center_y, x1, y1, x2, y2) < radius

    def _compute_edge_parameters(self, soc_i, soc_j, distance):
        """
        Computes feasibility, lambda (λ) value, and line segment parameters for an edge.
        
        Args:
            soc_i (float): State of charge at node i.
            soc_j (float): State of charge at node j.
            distance (float): Euclidean distance between node i and node j.
            
        Returns:
            tuple: (feasible (bool), lamb (float or None), line_segment (dict))
        """
        feasible = False
        line_segment = {}
        
        # Check if the SOC requirements immediately rule out feasibility
        if soc_j > min(self.q_max, soc_i + self.beta * distance) or (soc_j < soc_i - self.alpha * distance):
            return feasible, None, line_segment

        else:
            lamb_1 = (self.q_max - soc_i) / (self.beta * distance)
            lamb_2 = (self.q_max - soc_j) / (self.alpha * distance)
            
            if lamb_1 >= 0 and lamb_2 >= 0 and (lamb_1 + lamb_2 <= 1):
                lamb = lamb_1 + self.alpha / (self.beta + self.alpha) * (1 - lamb_1 - lamb_2)
                delta_SOC = self.beta / (self.beta + self.alpha) * (1 - lamb_1 - lamb_2) 
                if soc_j - delta_SOC < self.q_min:
                    # lamb_1_sw = lamb_1 + (soc_j - self.q_min) / (self.alpha * distance)
                    # lamb_2_sw = lamb_1_sw + (1- lamb_1_sw) * self.alpha / (self.beta + self.alpha)
                    feasible = False
                    return feasible, None, line_segment
                
                lamb_2_sw = lamb_2 + self.beta / (self.beta + self.alpha) * (1 - lamb_1 - lamb_2)
                
                # Note: using the key "g" twice in a dict will override; if multiple segments are needed,
                # consider using a list of segments.
                line_segment = {"g": (0, lamb_1), "e": (lamb_1, lamb_2_sw), "g2": (lamb_2_sw, 1)}
                
            else:
                lamb = (soc_j - soc_i + self.alpha * distance) / ((self.alpha + self.beta) * distance)
                line_segment = {"g": (0, lamb), "e": (lamb, 1)}
            if 0 <= lamb <= 1:
                feasible = True
                
        return feasible, lamb, line_segment

    def external_edge_addition(self, i, node_i, j, node_j, visibility_graph):
        """
        Adds an external edge to the visibility graph if the edge is feasible.
        
        Args:
            i (int): Node index for node_i.
            node_i (array-like): Information for node i.
            j (int): Node index for node_j.
            node_j (array-like): Information for node j.
            visibility_graph (networkx.Graph): The graph to update.
            
        Returns:
            networkx.Graph: The updated visibility graph.
        """
        distance = np.linalg.norm(np.array(node_i[:2]) - np.array(node_j[:2]))
        for i_add, SOC_i in enumerate(self.SOC):
            for j_add, SOC_j in enumerate(self.SOC):
                # print(f"i: {i+i_add}, j: {j+j_add}")
                feasible, lamb, line_segment = self._compute_edge_parameters(SOC_i, SOC_j, distance)
                # print(f"i: {i+i_add}, j: {j+j_add}")
                if feasible:
                    # print(f" after fesibility check is true i: {i+i_add}, j: {j+j_add}")
                    cost = lamb * distance
                    visibility_graph.add_edge(
                        i+i_add, j+j_add,
                        node_i_info= (node_i[0], node_i[1], node_i[2], SOC_i),
                        node_j_info= (node_j[0], node_j[1], node_j[2], SOC_j),
                        line_segment=line_segment,
                        fuel_cost=cost,
                        risk_cost=0,
                        feasibility=feasible,
                        edge_type="external"
                    )
        return visibility_graph

    def internal_edge_addition(self, i, node_i, j, node_j, visibility_graph):
        """
        Adds an internal edge (within a circle) to the visibility graph if the edge is feasible.
        Also computes a risk cost for the edge.
        
        Args:
            i (int): Node index for node_i.
            node_i (array-like): Information for node i.
            j (int): Node index for node_j.
            node_j (array-like): Information for node j.
            visibility_graph (networkx.Graph): The graph to update.
            
        Returns:
            networkx.Graph: The updated visibility graph.
        """
        distance = np.linalg.norm(np.array(node_i[:2]) - np.array(node_j[:2]))
        # Retrieve circle parameters corresponding to node_i (assumes node_i[2] is valid)
        circle = self.map_qz[int(node_i[2] - 1)]
        _, _, radius, radius_in, risk_limit, toggle = circle
        pen_dist = abs(sp.sqrt(radius**2 - (distance / 2)**2))
        
        for i_add, SOC_i in enumerate(self.SOC):
            for j_add, SOC_j in enumerate(self.SOC):                
                if pen_dist < radius_in and toggle==0:
                    feasible = False
                elif pen_dist < radius_in and toggle==1:
                    feasible, lamb, line_segment = self._compute_edge_parameters(SOC_i, SOC_j, distance)
                    if feasible:
                        r_minus, r_plus = 0.5 - radius_in/(2*radius), 0.5 + radius_in/(2*radius)
                        l_minus, l_plus = line_segment["e"]
                        electric_len_req = 2*sp.sqrt(radius_in**2 - pen_dist**2)
                        
                        if (l_plus-l_minus) >= electric_len_req and feasible:
                            if (l_minus <= r_minus) and (l_plus >= r_plus):
                                feasible = True
                            elif (l_minus >= r_minus) and (l_plus >= r_plus):
                                line_segment["g"] = (0, r_minus)
                                line_segment["e"] = (r_minus, r_minus+(l_plus-l_minus))
                                line_segment["g2"] = (r_minus+(l_plus-l_minus), 1)
                            else:
                                feasible = False
                        else:
                            feasible = False
                else:
                    feasible, lamb, line_segment = self._compute_edge_parameters(SOC_i, SOC_j, distance)
                    
                if feasible:
                    
                    cost = lamb * distance
                    # print(f"SOC_i: {SOC_i} and SOC_j: {SOC_j}, line_segment: {line_segment} , pen_dist: {pen_dist} and distance {distance}")
                    risk_cost = self._compute_risk_cost(line_segment, pen_dist, distance)
                    
                    visibility_graph.add_edge(
                        i+i_add, j+j_add,
                        node_i_info= (node_i[0], node_i[1], node_i[2], SOC_i),
                        node_j_info= (node_j[0], node_j[1], node_j[2], SOC_j),
                        line_segment=line_segment,
                        fuel_cost=cost,
                        risk_cost=risk_cost,
                        feasibility=feasible,
                        edge_type="internal"
                        )
        return visibility_graph

    def _compute_risk_cost(self, line_segment, pen_dist, distance):
        """
        Computes the risk cost for an internal edge based on the line segment parameters.
        
        Args:
            line_segment (dict): Contains segments with keys (e.g. "g", "e").
            pen_dist: Penetration distance (symbolic) computed from the circle.
            distance (float): Euclidean distance between the two nodes.
            
        Returns:
            risk_cost (sympy expression): The computed risk cost.
        """
        risk_cost = 0
        constant_factor = 30
        factor = 1 / (pen_dist + 0.01)
        
        for key, (lamb_start, lamb_end) in line_segment.items():
            if key == "e":
                continue  # 'e' segments do not contribute to risk cost
            elif key.startswith("g"):
                if lamb_start <= 0.5 <= lamb_end:
                    risk_cost += abs(
                        factor * (
                            sp.atan((distance/2 - lamb_start * distance) / (pen_dist + 0.01)) +
                            sp.atan((lamb_end * distance - distance/2) / (pen_dist + 0.01))
                        )
                    )
                elif lamb_start >= 0.5:
                    risk_cost += abs(
                        factor * (
                            sp.atan((lamb_end * distance - distance/2) / (pen_dist + 0.01)) -
                            sp.atan((lamb_start * distance - distance/2) / (pen_dist + 0.01))
                        )
                    )
                elif lamb_end <= 0.5:
                    risk_cost += abs(
                        factor * (
                            sp.atan((distance/2 - lamb_start * distance) / (pen_dist + 0.01)) -
                            sp.atan((distance/2 - lamb_end * distance) / (pen_dist + 0.01))
                        )
                    )
        return constant_factor * risk_cost

    def build_visibility_graph(self, rev_index_map):
        """
        Builds the visibility graph by adding internal and external edges between nodes.
        
        Args:
            graph_obj: Instance of GraphConstructionDiscretization.
            rev_index_map: Mapping of node indices to node information.
            
        Returns:
            visibility_graph: A directed graph (nx.DiGraph) with added edges.
        """
        self.visibility_graph = nx.DiGraph()
        # Iterate over all pairs of nodes
        for i in range(0, len(rev_index_map), self.num_SOC):
            for j in range(0, len(rev_index_map), self.num_SOC):
                if i == j or i == len(rev_index_map) - 1 or j == len(rev_index_map) - 1:
                    continue
                node_i, node_j = rev_index_map[i], rev_index_map[j]
                
                circle_i, circle_j = node_i[2], node_j[2]
                x_i, y_i = node_i[0], node_i[1]
                x_j, y_j = node_j[0], node_j[1]

                # Case 1: Internal edge if both nodes are from the same circle
                if circle_i == circle_j:
                    self.internal_edge_addition(i, node_i, j, node_j, self.visibility_graph)
                else:
                    # For external edges, check if any QZ circle intersects the line segment
                    intersection_exists = False
                    for circle in self.map_qz:
                        if self.line_segment_intersects_circle(circle, (x_i, y_i), (x_j, y_j)):
                            intersection_exists = True
                            break
                    if not intersection_exists:
                        self.external_edge_addition(i, node_i, j, node_j, self.visibility_graph)

        ### Assign the and cost from goal to dummy node as zero 
        for i in range(self.num_SOC , 2*self.num_SOC):
            j = len(rev_index_map) - 1
            node_i, node_j = rev_index_map[i], rev_index_map[j]
            
            self.visibility_graph.add_edge(i, j,
                        node_i_info= (node_i[0], node_i[1], node_i[2], node_i[3]),
                        node_j_info= (node_j[0], node_j[1], node_j[2], node_j[3]),
                        line_segment= None,
                        fuel_cost= 0,
                        risk_cost= 0,
                        feasibility= True,
                        edge_type="external"
                    )
            self.visibility_graph.add_edge(j, i,
                node_j_info= (node_j[0], node_j[1], node_j[2], node_j[3]),
                node_i_info= (node_i[0], node_i[1], node_i[2], node_i[3]),
                line_segment= None,
                fuel_cost= 0,
                risk_cost= 0,
                feasibility= True,
                edge_type="external"
            )
        
    def assign_heuristic_costs(self, rev_index_map):
        """
        Assigns a heuristic cost for each node based on its Euclidean distance to the goal.
        
        Args:
            rev_index_map: Mapping from node index to node tuple.
        """
        # Define the goal node tuple. The circle index for the goal is set to len(qz_circles)+1. Assume that the goal is reached at q_min. 
        # In this way, the heuristic cost is the lowest possible.
        goal_node = (self.goal[0], self.goal[1], len(self.map_qz) + 1, self.q_min)
        
        for i, node in rev_index_map.items():
            if node[2] != len(self.map_qz) + 1:
                # Compute Euclidean distance using sympy (for consistency with the rest of the code)
                distance_goal = sp.sqrt((goal_node[0] - node[0])**2 + (goal_node[1] - node[1])**2)

                heuristic_cost_forward = max( ( self.alpha*distance_goal + goal_node[3] - node[3]) / (self.alpha + self.beta) , 0 )            
                self.visibility_graph.nodes[i]['heuristic_cost'] = {"fuel_cost": heuristic_cost_forward, "risk_cost": 0}
            else:
                # print(f" The nodes that are assigned hueristic cost of zero are {node}")
                self.visibility_graph.nodes[i]['heuristic_cost'] = {"fuel_cost": 0, "risk_cost": 0}

class Biobjective_search_and_heuristic_calc_class():
    """
    Contains functions for heuristic calculation and biobjective search 
    """
    def __init__(self, graph_object, start_state, goal_state):
        self.graph_object = graph_object
        self.start_state = start_state
        self.goal_state = goal_state
    
    def update_heuristic(self, state, heuristic, Cost_type="fuel_cost"):
        # Thread-safe update of an upper bound.
        # print(f"Updating heuristic for state, state is {state}, {self.graph_object.visibility_graph.nodes[state]}")
        self.graph_object.visibility_graph.nodes[state]["heuristic_cost"][Cost_type] = heuristic

    ### Cost_Bounded_forward_Astar
    ### Cost_type = 1 for fuel cost, 2 for risk cost
    def Cost_Bounded_backward_Astar_f2f1(self): 
        # Initialize the priority queue with the start state.
        open_list = []
        closed = []
        f2min = {node: np.inf for node in self.graph_object.visibility_graph.nodes()}    
        ### node is of the form (f1, g1, f2, g2, state)
        heapq.heappush(open_list, (0, 0, 0, 0, self.goal_state))
        
        while open_list:
            current_f2, current_f1, current_g2, current_g1, current_state = heapq.heappop(open_list)
            # print(f"current state: {current_state}, current_f2: {current_f2}, current_g2: {current_g2}, current_f1: {current_f1}, current_g1: {current_g1}")
            if current_state not in closed:
                ### Check if the current state is the goal state
                if current_state == self.start_state:
                    self.update_heuristic(current_state, current_g2, Cost_type="risk_cost")
                    f2min[current_state] = current_g2
                    closed.append(current_state)
                    continue
                
                self.update_heuristic(current_state, current_g2, Cost_type="risk_cost")
                
                ### Expand successors
                for successor in self.graph_object.visibility_graph.successors(current_state):
                    if successor not in closed:
                        edge_data = self.graph_object.visibility_graph.edges[current_state, successor]
                        node_data = self.graph_object.visibility_graph.nodes[successor]
                        
                        successor_g1 = current_g1 + edge_data["fuel_cost"]
                        successor_g2 = current_g2 + edge_data["risk_cost"]
                        
                        successor_f1 = successor_g1 + node_data["heuristic_cost"]["fuel_cost"]
                        successor_f2 = successor_g2 + node_data["heuristic_cost"]["risk_cost"]
                        
                        ### The successor is in open list, but its value less than the current g1 value, update the g1min value and put it in the open list
                        if successor_f2 < f2min[successor]:
                            f2min[successor] = successor_f2
                            heapq.heappush(open_list, ( successor_f2, successor_f1, successor_g2, successor_g1, successor))
                        
                closed.append(current_state)

    def Cost_Bounded_backward_Astar_f1f2(self): 
        # Initialize the priority queue with the start state.
        open_list = []
        closed = []
        f1min = {node: np.inf for node in self.graph_object.visibility_graph.nodes()}    
        ### node is of the form (f1, g1, f2, g2, state)
        heapq.heappush(open_list, (0, 0, 0, 0, self.goal_state))
        
        while open_list:
            current_f1, current_f2, current_g1, current_g2, current_state = heapq.heappop(open_list)
            # print(f"current state: {current_state}, current_f2: {current_f2}, current_g2: {current_g2}, current_f1: {current_f1}, current_g1: {current_g1}")
            if current_state not in closed:
                ### Check if the current state is the goal state
                if current_state == self.start_state:
                    self.update_heuristic(current_state, current_g1, Cost_type="fuel_cost")
                    f1min[current_state] = current_g1
                    closed.append(current_state) 
                    continue
                
                self.update_heuristic(current_state, current_g1, Cost_type="fuel_cost")
                
                ### Expand successors
                for successor in self.graph_object.visibility_graph.successors(current_state):
                    if successor not in closed:
                        edge_data = self.graph_object.visibility_graph.edges[current_state, successor]
                        node_data = self.graph_object.visibility_graph.nodes[successor]
                        
                        successor_g1 = current_g1 + edge_data["fuel_cost"]
                        successor_g2 = current_g2 + edge_data["risk_cost"]
                        
                        successor_f1 = successor_g1 + node_data["heuristic_cost"]["fuel_cost"]
                        successor_f2 = successor_g2 + node_data["heuristic_cost"]["risk_cost"]

                        ### The successor is in open list, but its value less than the current g1 value, update the g1min value and put it in the open list
                        if successor_f1 < f1min[successor]:
                            f1min[successor] = successor_f1
                            heapq.heappush(open_list, ( successor_f1, successor_f2, successor_g1, successor_g2, successor))
                        
                closed.append(current_state)

    def Heuristic_calc(self):
        print("Calculating heuristic")
        self.Cost_Bounded_backward_Astar_f1f2()
        self.Cost_Bounded_backward_Astar_f2f1()
    
    def reduce_factor_calc(self, state, current_f1, f1_min, type="constant", reduce_factor_constant = 1):
        if type == "variable":
            if f1_min[state] != 0:
                reduce_factor = (1 - 5*(current_f1 - f1_min[state])/current_f1)
            else:
                reduce_factor = 1
            
            return reduce_factor
        elif type == "constant":
            reduce_factor = reduce_factor_constant
        return reduce_factor
    
    def biobjective_search( self, type = "constant", reduce_factor_constant=0.9):
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
        visibility_graph = self.graph_object.visibility_graph
        max_risk_limit, acceptable_risk_limit = self.graph_object.max_risk_limit, self.graph_object.acceptable_risk_limit

        all_states = list(visibility_graph.nodes)
        # print("All states:", all_states)
        sols = {state: [] for state in all_states}
        g2_min = {state: np.inf for state in all_states}
        f1_min = {state: 0 for state in all_states}
        open_set = []

        # The start node is represented as a tuple: (f1, f2, g1, g2, state)
        # and its parent is set to None.
        start_node = (0, 0, 0, 0, self.start_state)
        heapq.heappush(open_set, [start_node, (None, None, None, None, None)])

        while open_set:
            current_node, parent_node = heapq.heappop(open_set)
            current_f1, current_f2, current_g1, current_g2, current_state = current_node

            # Prune if the current risk cost is dominated
            reduce_factor = self.reduce_factor_calc(current_state, current_f1, f1_min, type="variable", reduce_factor_constant=reduce_factor_constant)
            if (current_g2 >= reduce_factor*g2_min[current_state] or
                current_f2 >= reduce_factor*g2_min[self.goal_state]):
                continue
            
            g2_min[current_state] = current_g2
            f1_min[current_state] = current_f1
            sols[current_state].append([current_node, parent_node])

            if current_state == "g" and current_g2 < acceptable_risk_limit:
                return sols, g2_min
            
            # Stop expanding if the goal is reached
            if current_state == self.goal_state:
                print(f"What is the current state: {current_state}, current f1 {current_f1}, current f2 {current_f2}, current g1 {current_g1}, current g2 {current_g2}")
                continue

            # Expand successors of the current state
            for successor in visibility_graph.successors(current_state):
                edge_data = visibility_graph.edges[current_state, successor]
                g1 = current_g1 + edge_data['fuel_cost']
                f1 = g1 + visibility_graph.nodes[successor]['heuristic_cost']["fuel_cost"]
                g2 = current_g2 + edge_data['risk_cost']
                # For risk, we use a zero heuristic.
                f2 = g2 + visibility_graph.nodes[successor]['heuristic_cost']["risk_cost"]

                # Prune dominated successors
                reduce_factor = self.reduce_factor_calc(successor, f1, f1_min, type="variable", reduce_factor_constant=reduce_factor_constant)
                reduce_factor_goal = self.reduce_factor_calc(self.goal_state, f1, f1_min, type="variable", reduce_factor_constant=reduce_factor_constant)
                
                if (g2 >= reduce_factor * g2_min[successor] or
                    f2 >= reduce_factor_goal * g2_min[self.goal_state]) or (current_g2 > max_risk_limit):
                    continue

                child_node = (f1, f2, g1, g2, successor)
                heapq.heappush(open_set, [child_node, current_node])
        
        return sols, g2_min

def extract_costs(solutions, target_state):
    """
    Extracts fuel and risk cost values for a given target state from the solutions.
    
    Args:
        solutions: Dict mapping states to solution tuples.
        target_state: The state for which to extract cost values.
        
    Returns:
        fuel_costs: List of fuel cost values.
        risk_costs: List of risk cost values.
    """
    fuel_costs = [sol[0][0] for sol in solutions[target_state]]
    risk_costs = [sol[0][1] for sol in solutions[target_state]]
    return fuel_costs, risk_costs


def check_pareto_optimality(fuel_costs, risk_costs):
    """
    Checks for dominance among the solutions and returns the indices of dominated solutions.
    
    Args:
        fuel_costs: List of fuel cost values.
        risk_costs: List of risk cost values.
        
    Returns:
        dominated_indices: List of indices corresponding to dominated solutions.
    """
    dominated_indices = []
    n = len(fuel_costs)
    for i in range(n):
        for j in range(i + 1, n):
            if fuel_costs[i] <= fuel_costs[j] and risk_costs[i] <= risk_costs[j]:
                dominated_indices.append(i)
            elif fuel_costs[i] >= fuel_costs[j] and risk_costs[i] >= risk_costs[j]:
                dominated_indices.append(j)
    return dominated_indices

def plot_costs(fuel_costs, risk_costs):
    """
    Plots a scatter plot of fuel cost versus risk cost.
    
    Args:
        fuel_costs: List of fuel cost values.
        risk_costs: List of risk cost values.
    """
    plt.figure(figsize=(14, 10))
    plt.scatter(fuel_costs, risk_costs, color='royalblue', edgecolors='black',
                alpha=0.75, s=80)
    plt.xlabel("Fuel Cost", fontsize=14, fontweight='bold')
    plt.ylabel("Risk Cost", fontsize=14, fontweight='bold')
    plt.title("Fuel Cost vs. Risk Cost", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    
# =============================================================================
# Reconstructing Solution Paths
# =============================================================================
def reconstruct_solution_paths(solutions, start_state="s", goal_state="g"):
    """
    Reconstructs paths from the start state to the goal state using the parent links stored in solutions.
    
    Each solution is a tuple of the current node and its parent. The node tuple is in the form:
        (f1, f2, g1, g2, state)
    
    Returns:
        solution_paths: A list of paths. Each path is a list of states followed by a cost tuple.
    """
    solution_paths = []
    # Iterate over each solution for the goal state
    for current_node, parent_node in solutions[goal_state]:
        path = [current_node[-1]]  # start with the goal state
        cost = (current_node[0], current_node[1])
        # Traverse backward through the parent links
        parent_state = parent_node[-1] if parent_node[4] is not None else None
        curr_node, par_node = current_node, parent_node
        while parent_state is not None and parent_state != start_state:
            found = False
            for sol in solutions[parent_state]:
                if sol[0] == par_node:
                    curr_node, par_node = sol
                    path.append(curr_node[-1])
                    parent_state = par_node[-1]
                    found = True
                    break
            if not found:
                break
        if par_node is not None:
            path.append(par_node[-1])
        path.reverse()
        # Append the associated cost tuple at the end of the path
        path.append(cost)
        solution_paths.append(path)
    return solution_paths

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

def plot_map_with_path_and_soc(graph_object, Map_qz, start, goal, path, reverse_index_map):
    # Create 2 subplots with equal heights in a single figure.
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    ##################################################
    # TOP SUBPLOT: The map with the path and zones
    ##################################################
    
    # Find min and max x and y for proper axis limits
    min_x = min([
        min(Map_qz, key=lambda x: x[0]-x[2])[0] - min(Map_qz, key=lambda x: x[0]-x[2])[2],
        start[0], goal[0]
    ])
    max_x = max([
        max(Map_qz, key=lambda x: x[0]+x[2])[0] + max(Map_qz, key=lambda x: x[0]+x[2])[2],
        start[0], goal[0]
    ])
    min_y = min([
        min(Map_qz, key=lambda x: x[1]-x[2])[1] - min(Map_qz, key=lambda x: x[1]-x[2])[2],
        start[1], goal[1]
    ])
    max_y = max([
        max(Map_qz, key=lambda x: x[1]+x[2])[1] + max(Map_qz, key=lambda x: x[1]+x[2])[2],
        start[1], goal[1]
    ])
    
    # For path segmentation, we prepare two lists for colored segments
    path_segments_gas = []
    path_segments_electric = []

    # Iterate over the path segments to separate them by mode.
    # In our case, we use the mode of the second point in the segment.
    for i in range(len(path) - 1):
        pt1 = path[i]
        pt2 = path[i+1]
        # Determine segment based on the mode of the second point in the segment.
        if pt2[3] == 'g':
            path_segments_gas.append((pt1, pt2))
        else:
            path_segments_electric.append((pt1, pt2))

    # Plot gas segments in red and electric segments in green
    for seg in path_segments_gas:
        x_vals = [seg[0][0], seg[1][0]]
        y_vals = [seg[0][1], seg[1][1]]
        ax1.plot(x_vals, y_vals, 'r', linewidth=2, label='Gas' if 'Gas' not in [l.get_label() for l in ax1.lines] else "")
    for seg in path_segments_electric:
        x_vals = [seg[0][0], seg[1][0]]
        y_vals = [seg[0][1], seg[1][1]]
        ax1.plot(x_vals, y_vals, 'g', linewidth=2, label='Electric' if 'Electric' not in [l.get_label() for l in ax1.lines] else "")

    # Draw the zones using circles
    for circle_info in Map_qz:
        # Outer circle (semi-transparent blue)
        circle_outer = Circle(
            (circle_info[0], circle_info[1]), 
            radius=circle_info[2],
            fill=True, 
            facecolor=(0, 0, 1, 0.2), 
            edgecolor='blue', 
            linewidth=2, 
            zorder=1
        )
        # Inner circle (darker)
        circle_inner = Circle(
            (circle_info[0], circle_info[1]), 
            radius=circle_info[3],
            fill=True, 
            facecolor=(0, 0, 0, 0.5),
            edgecolor='black', 
            linewidth=2, 
            zorder=2
        )
        ax1.add_patch(circle_outer)
        ax1.add_patch(circle_inner)

    # Plot start and goal points
    ax1.plot(start[0], start[1], 'ro', markersize=10, label='Start')
    ax1.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')

    # Set axes properties for map
    ax1.set_xlim(min_x - (max_x - min_x)/10, max_x + (max_x - min_x)/10)
    ax1.set_ylim(min_y - (max_y - min_y)/10, max_y + (max_y - min_y)/10)
    ax1.set_aspect('equal', 'box')
    ax1.set_title("Pareto Optimal Paths")
    ax1.grid(True)
    ax1.legend()

    ##################################################
    # BOTTOM SUBPLOT: SOC vs. Distance traveled, with colors by mode.
    ##################################################
    
    # Compute cumulative distance traveled for each point.
    cumulative_distance = [0.0]  # starting with distance 0 at the first point
    for i in range(1, len(path)):
        x_prev, y_prev = path[i-1][0], path[i-1][1]
        x_curr, y_curr = path[i][0], path[i][1]
        seg_distance = math.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
        cumulative_distance.append(cumulative_distance[-1] + seg_distance)

    # Plot each SOC segment according to the mode.
    # We use the same segmentation as above: for the segment from i to i+1, the mode is in path[i+1][3].
    for i in range(len(path)-1):
        d_start = cumulative_distance[i]
        d_end = cumulative_distance[i+1]
        soc_start = path[i][2]
        soc_end = path[i+1][2]
        
        # Determine color based on mode at the second point of the segment.
        color = 'r' if path[i+1][3] == 'g' else 'g'
        # Plot the line segment with markers.
        ax2.plot([d_start, d_end], [soc_start, soc_end], color=color, linewidth=2,
                 marker='o', markersize=5,
                 label='Gas' if (color=='r' and 'Gas' not in [line.get_label() for line in ax2.lines])
                        else ('Electric' if (color=='g' and 'Electric' not in [line.get_label() for line in ax2.lines])
                              else ""))
        
    # Set labels and title for the SOC plot
    ax2.set_xlabel("Distance Traveled")
    ax2.set_ylabel("SOC")
    ax2.set_title("SOC vs. Distance Traveled")
    ax2.grid(True)

    # Adjust layout to ensure both subplots are of equal size and well spaced.
    plt.tight_layout()
    plt.show()

# Example usage:
# graph_object, Map_qz, start, goal, path, reverse_index_map = ... (define these)
# plot_map_with_path_and_soc(graph_object, Map_qz, start, goal, path, reverse_index_map)

