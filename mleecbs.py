import time as timer
import heapq
import random
import copy
import time
from focalsearch_single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
import csv
import joblib
import matplotlib.pyplot as plt
import warnings
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, node_colors=None):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter, node_colors)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, node_colors=None, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                 node_colors=node_colors, pos=pos, parent=root, parsed=parsed)
    return pos



# Suppress all warnings
warnings.filterwarnings('ignore')

def detect_first_collision_for_path_pair(path1, path2):
    ##############################
    # Task 2.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    collision = None
    min_len = min(len(path1), len(path2))
    for i in range(max(len(path1), len(path2))-1):
        loc1_1, loc1_2, loc2_1, loc2_2 = get_location(path1,i), get_location(path1,i+1), get_location(path2,i), get_location(path2,i+1)
        if loc1_1 == loc2_1:
            collision = {'loc': [loc1_1], 'timestep': i}
            return collision
        if loc1_1 == loc2_2 and loc1_2 == loc2_1:
            collision = {'loc': [loc1_1, loc1_2], 'timestep': i+1}
            return collision

    if path1[-1] == path2[-1]:
            collision = {'loc': [path1[-1]], 'timestep': i}
            return collision
    return None


def detect_collisions_among_all_paths(paths):
    ##############################
    # Task 2.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    collisions = []
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            coll_dict = {'a1': i, 'a2': j}
            collision = detect_first_collision_for_path_pair(paths[i], paths[j])
            if collision:
                coll_dict.update(collision)
                collisions.append(coll_dict)
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 2.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    constraints = []
    c1 = {'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']}
    c2 = {'agent': collision['a2'], 'loc': collision['loc'][::-1], 'timestep': collision['timestep']}
    constraints.append(c1)
    constraints.append(c2)
    return constraints




class MLEECBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals, map_name, batch):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.counter = 0
        self.svr = joblib.load('svm_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        with open('feature_names.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            self.feature_names = next(reader)

        self.open_list = []

        self.focal_list = []
        # Define the tree structure
        self.tree = nx.DiGraph()
        self.cleanup_list = []
        self.closed_dict = {}
        self.tree_path = []
        self.node_num = 0
        self.epsbar_d = 0
        self.epsbar_h = 0
        self.h_hatprime = 0 # h_hat without h_c
        self.w = 1.2
        self.onestep_count = 0
        self.viz = not batch
        self.map_name = map_name
        self.save_file = 'mlresults_' + map_name + '.csv'

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def update_h_hatprime(self):
        self.h_hatprime = max(min(self.epsbar_h/(1-self.epsbar_d + 1e-8), 500),-500)

    def push_node(self, node):
        heapq.heappush(self.cleanup_list, (node['LB'], len(node['collisions']), self.num_of_generated, node))
        heapq.heappush(self.open_list, (node['f_hat'], len(node['collisions']), self.num_of_generated, node))
        if node['f_hat']<=self.w*self.open_list[0][-1]['f_hat']:
            heapq.heappush(self.focal_list, (len(node['collisions']), node['f_hat'], self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        self.num_of_expanded+=1
        if len(self.focal_list)>0 and self.focal_list[0][-1]['cost'] <= self.w * self.cleanup_list[0][-1]['LB']:
            _, _, id, node = heapq.heappop(self.focal_list)
            for idx, ( _, _, _, a_dict) in enumerate(self.open_list):
                if a_dict is node:  
                    del self.open_list[idx]  
                    break
            
            for idx, ( _, _, _, a_dict) in enumerate(self.cleanup_list):
                if a_dict is node:  
                    del self.cleanup_list[idx]  
                    break
            print("Expand node {} from FOCAL".format(id))
            return node
        
        elif self.open_list[0][-1]['cost'] <= self.w * self.cleanup_list[0][-1]['LB']:
            _, _, id, node = heapq.heappop(self.open_list)
            for idx, ( _, _, _, a_dict) in enumerate(self.focal_list):
                if a_dict is node:  
                    del self.focal_list[idx]  
                    break
            
            for idx, ( _, _, _, a_dict) in enumerate(self.cleanup_list):
                if a_dict is node:  
                    del self.cleanup_list[idx]  
                    break
            print("Expand node {} from OPEN".format(id))
            return node
        
        else:
            _, _, id, node = heapq.heappop(self.cleanup_list)
            for idx, (  _, _, _, a_dict) in enumerate(self.open_list):
                if a_dict is node:  
                    del self.open_list[idx]  
                    break
            
            for idx, ( _, _, _, a_dict) in enumerate(self.focal_list):
                if a_dict is node:  
                    del self.focal_list[idx]  
                    break
            print("Expand node {} from CLEANUP".format(id))
            return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """
        lb_list = []
        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'LB':0,
                'f_hat':0,
                'constraints': [],
                'paths': [],
                'collisions': [],
                'node_num':0,
                'parent': None,
                'h_hat':0,
                'h_ml':0,
                'num_coll':0,
                'node_level':0,
                'num_agents':self.num_of_agents,
                'nlvl/num': 0}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path, lb = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'], other_paths=root['paths'], w=self.w)
            lb_list.append(lb)
            # self.counter+=1
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)
        root['LB']=sum(lb_list)
        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'])
        root['num_coll'] = len(root['collisions'])
        x = self.scaler.transform([[root[feature_name] for feature_name in self.feature_names]])
        root['f_hat'] = root['cost'] + self.svr.predict(x)
        root['h_ml'] = self.svr.predict(x)
        
        self.push_node(root)

        self.tree.add_node(str(id(root)), value = self.num_of_generated)

        # Task 2.1: Testing
        print(root['collisions'])

        # Task 2.2: Testing
        for collision in root['collisions']:
            print(standard_splitting(collision))

        ##############################
        # Task 2.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        # These are just to print debug output - can be modified once you implement the high-level search
        while self.open_list:
            curr = self.pop_node()
            if curr != root:
                self.tree.add_node(str(id(curr)), value = self.num_of_generated)
                self.tree.add_edge(str(curr['parent']),str(id(curr)))
            if  len(curr['collisions'])==0:
                self.print_results(curr)
                self.write_results(curr)
                # fieldnames = ['cost','LB','f_hat','cost-to-go', 'h_hat']
                # csv_file = 'data_collection.csv'
                # Get numerical values of nodes
                if self.viz:
                    print(list(self.tree.nodes.keys()))
                    node_values = {node: self.tree.nodes[node]["value"] for node in self.tree.nodes}

                    # Define colormap
                    colormap = plt.cm.viridis

                    # Normalize values to range [0, 1] for colormap
                    norm = Normalize(vmin=min(node_values.values()), vmax=max(node_values.values()))

                    # Create a ScalarMappable to map numerical values to colors
                    sm = ScalarMappable(norm=norm, cmap=colormap)
                    sm.set_array([])  # Dummy array for ScalarMappable

                    # Get node colors based on their numerical values
                    node_colors = {node: sm.to_rgba(value) for node, value in node_values.items()}

                    # Visualize the tree
                    pos = hierarchy_pos(self.tree, str(id(root)), node_colors=node_colors)
                    nx.draw(self.tree, pos, with_labels=True, node_size=10, node_color=list(node_colors.values()), font_size=0.1, arrows=False)
                    # plt.colorbar(sm, label="Node Value")
                    plt.title("Tree Visualization")
                    plt.show()
                    
                tree_path = self.backtrack(curr)
                y_true = []
                y_heur = []
                y_ml = []
                for node in tree_path:
                    y_true.append(curr['cost'] - node['cost'])
                    y_heur.append(node['h_hat'])
                    y_ml.append(node['h_ml'])
                plt.scatter(y_true, y_ml, color='blue', alpha=0.2, label = "ML-EECBS")
                plt.scatter(y_true, y_heur, color='green', alpha=0.2, label = "EECBS")
                plt.plot([0, max(y_true)], [0, max(y_true)], color='red', linestyle='--')
                plt.xlim([0, max(y_true)])
                plt.ylim([0, max(y_true)])
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Actual vs. Predicted Values ({self.map_name}_{self.num_of_agents})')
                plt.legend()
                if self.viz:
                    plt.show()
                plt.savefig(f'SVR_Results_{self.map_name}_{self.num_of_agents}.png')
                plt.close()
                # print("Writing to file", csv_file, "\n")
                # for row in tree_path:
                #     row['cost-to-go'] = curr['cost'] - row['cost']
                # with open(csv_file, mode='w', newline='') as file:
                #     writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                #     # Write header
                #     writer.writeheader()
                    
                #     # Write rows
                #     for row in tree_path:
                #         writer.writerow({fieldname: row[fieldname] for fieldname in fieldnames})
                return curr['paths']
            
            collision = curr['collisions'][0]
            
 
            constraints = standard_splitting(collision)
            children = []

            for constraint in constraints:
                child = copy.deepcopy(curr)
                self.node_num +=1
                child['constraints'].append(constraint)
                agent = constraint['agent']
                other_paths = child['paths'][:agent] + child['paths'][agent+1:]
  
                path, lb_list[agent] = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent],
                            agent, child['constraints'], other_paths, w=self.w)
                child['LB'] = sum(lb_list)
                
                if path is None:
                    continue

                child['paths'][agent] = path
                child['cost'] = get_sum_of_cost(child['paths'])
                child['collisions'] = detect_collisions_among_all_paths(child['paths'])
                child['node_level'] = curr['node_level'] + 1
                child['num_coll'] = len(child['collisions'])
                child['node_num'] = self.node_num
                child['nlvl/num'] = child['node_level']/self.num_of_agents
                child['parent'] = id(curr)
                
                child['h_hat'] = self.h_hatprime*len(child['collisions'])

                x = self.scaler.transform([[child[feature_name] for feature_name in self.feature_names]])
                child['f_hat'] = child['cost'] + self.svr.predict(x)
                child['h_ml'] = self.svr.predict(x)
                
                children.append(child)
                # print("New Collisions", child['collisions'])

                self.push_node(child)
                
            # print('')
            if children[0]['f_hat']<children[1]['f_hat']:
                self.update_errors(children[0], curr)
            elif children[0]['f_hat']>children[1]['f_hat']:
                self.update_errors(children[1], curr)
            else:
                if len(children[0]['collisions'])<len(children[1]['collisions']):
                    self.update_errors(children[0], curr)
                else:
                    self.update_errors(children[1], curr)
            self.closed_dict[id(curr)] = curr
        self.print_results(root)
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! \n")
        self.CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))

    def write_results(self, node):
        file_exists = os.path.isfile(self.save_file)
        with open(self.save_file, mode='a+' if file_exists else 'w', newline='') as file:
            fieldnames = ['num_agents', 'cost', 'time', 'num_nodes_gen', 'num_nodes_exp']
            data = [{'num_agents':self.num_of_agents,
                    'cost':get_sum_of_cost(node['paths']), 
                    'time':self.CPU_time, 
                    'num_nodes_gen':self.num_of_generated,
                    'num_nodes_exp': self.num_of_expanded}]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for row in data:
                writer.writerow(row)

    def update_errors(self, best_child, parent):
        self.onestep_count+=1
        eps_h = best_child['cost'] - parent['cost']
        eps_d = len(best_child['collisions']) - (len(parent['collisions']) - 1)
        self.epsbar_h = (self.epsbar_h * (self.onestep_count-1) + eps_h)/self.onestep_count
        self.epsbar_d = (self.epsbar_d * (self.onestep_count-1) + eps_d)/self.onestep_count
        self.update_h_hatprime()
    #     if self.h_hatprime<0:
    #         time.sleep(5)

    def backtrack(self, solution_node):
        path = []
        curr_node = solution_node
        while curr_node is not None:
            path.append(curr_node)
            parent = curr_node['parent']
            if parent is not None:
                curr_node = self.closed_dict.get(parent)
            else:
                curr_node = None
        path.reverse()
        return path
