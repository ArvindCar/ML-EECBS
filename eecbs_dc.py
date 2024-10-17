import time as timer
import heapq
import random
import copy
import time
from focalsearch_single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
import csv
import os


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




class EECBS_DC(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals, map_name):
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
        

        self.open_list = []

        self.focal_list = []
        self.cleanup_list = []
        self.closed_dict = {}
        self.tree_path = []
        self.node_num = 0
        self.epsbar_d = 0
        self.epsbar_h = 0
        self.h_hatprime = 0 # h_hat without h_c
        self.w = 1.2
        self.onestep_count = 0
        self.num_sol = 0
        self.max_sol = 8
        self.file_path = f'data_collection_{map_name}.csv'
        self.save_file = 'results_' + map_name + '.csv'
        print(self.file_path)

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def update_h_hatprime(self):
        self.h_hatprime = max(min(self.epsbar_h/(1-self.epsbar_d + 1e-8), 500),-500)
        # if abs(self.h_hatprime)==500:
        #     print("WTFFF")
        #     time.sleep(3)

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
        root = {'id':None,
                'cost': 0,
                'LB':0,
                'f_hat':0,
                'constraints': [],
                'paths': [],
                'collisions': [],
                'node_num':0,
                'parent': None,
                'h_hat':0,
                'node_level':0
                }
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
        root['f_hat'] = root['cost'] + self.h_hatprime*len(root['collisions'])
        root['id'] = id(root)
        
        self.push_node(root)

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
            if  len(curr['collisions'])==0:
                self.num_sol += 1
                self.print_results(curr)
                self.write_results(curr)
                fieldnames = ['id', 'cost', 'LB', 'f_hat', 'num_coll', 'cost-to-go', 'h_hat', 'node_level', 'nlvl/num', 'num_agents', 'parent']
                tree_path = self.backtrack(curr)
                print("Writing to file", self.file_path, "\n")
                for row in tree_path:
                    row['cost-to-go'] = curr['cost'] - row['cost']
                    row['num_coll'] = len(row['collisions'])
                    row['nlvl/num'] = curr['node_level']/self.num_of_agents
                    row['num_agents'] = self.num_of_agents
                # if self.num_sol == 1:

                if os.path.exists(self.file_path):
                    with open(self.file_path, mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                    
                        # Write header
                        # writer.writeheader()
                        print("Root: ", tree_path[0]['id'])
                        # Write rows
                        for row in tree_path:
                            writer.writerow({fieldname: row[fieldname] for fieldname in fieldnames})
                else:
                    with open(self.file_path, mode='w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                    
                        # Write header
                        writer.writeheader()
                        print("Root: ", tree_path[0]['id'])
                        # Write rows
                        for row in tree_path:
                            writer.writerow({fieldname: row[fieldname] for fieldname in fieldnames})


                # elif self.num_sol<self.max_sol:
                #     num_updates = 0
                #     with open(csv_file, mode='r', newline='') as file:
                #         reader = csv.DictReader(file)
                #         existing_rows = list(reader)
                        
                #     with open(csv_file, mode='a', newline='') as file:
                #         writer = csv.DictWriter(file, fieldnames=fieldnames)
                    
                #         # Write header
                #         writer.writeheader()
                #         print("Root: ", existing_rows[0]['id'])
                #         # Write rows
                #         for row in tree_path:
                #             row_exists = False
                            
                #             for existing_row in existing_rows:


                #                 if int(existing_row['id']) == int(row['id']):
                #                     row_exists = True
                #                     # Update the row with lower value if necessary
                #                     print(float(row['cost-to-go'])-float(existing_row['cost-to-go']))
                #                     if float(row['cost-to-go']) < float(existing_row['cost-to-go']):
                #                         existing_row.update({key: value for key, value in row.items() if key in fieldnames})
                #                         num_updates +=1
                #                     break
                            
                #             # If row doesn't exist, write it
                #             if not row_exists:
                #                 writer.writerow({fieldname: row[fieldname] for fieldname in fieldnames})
                #                 print(f"Number of updates in {csv_file}: {num_updates}")
                #             # If row exists, write the updated row
                #             else:
                #                 writer.writerow(existing_row)
                #     with open(csv_file, mode='r', newline='') as file:
                #         reader = csv.reader(file)
                #         num_rows = sum(1 for _ in reader)
                #         print(f"Number of rows in {csv_file}: {num_rows}")
                # else:
                return curr['paths']
                # return curr['paths']
                # continue
            collision = curr['collisions'][0]
            constraints = standard_splitting(collision)
            children = []
            # print('hey',constraints)

            for constraint in constraints:
                child = copy.deepcopy(curr)
                self.node_num +=1
                child['constraints'].append(constraint)
                # print('Constraint:', child['constraints'])
                agent = constraint['agent']
                other_paths = child['paths'][:agent] + child['paths'][agent+1:]
                # print(other_paths)
                # time.sleep(3)
                path, lb_list[agent] = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent],
                            agent, child['constraints'], other_paths, w=self.w)
                child['LB'] = sum(lb_list)
                
                if path is None:
                    continue

                child['paths'][agent] = path
                child['cost'] = get_sum_of_cost(child['paths'])
                child['collisions'] = detect_collisions_among_all_paths(child['paths'])
                child['f_hat'] = child['cost'] + self.h_hatprime*len(child['collisions'])
                child['node_num'] = self.node_num
                child['parent'] = id(curr)
                child['h_hat'] = self.h_hatprime*len(child['collisions'])
                child['node_level'] = curr['node_level'] + 1
                child['id'] = id(child)
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
            # self.closed_list.append(curr)
            self.closed_dict[id(curr)] = curr
        self.print_results(root)
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! Soln No. ", self.num_sol, "\n")
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

            # Write header only if the file is newly created
            if not file_exists:
                writer.writeheader()

            # Write data to the file
            for row in data:
                # print("Writing: ", row)
                writer.writerow(row)
                # time.sleep(3)

    def update_errors(self, best_child, parent):
        self.onestep_count+=1
        eps_h = best_child['cost'] - parent['cost']
        eps_d = len(best_child['collisions']) - (len(parent['collisions']) - 1)
        self.epsbar_h = (self.epsbar_h * (self.onestep_count-1) + eps_h)/self.onestep_count
        self.epsbar_d = (self.epsbar_d * (self.onestep_count-1) + eps_d)/self.onestep_count
        self.update_h_hatprime()
        # print(self.h_hatprime, self.epsbar_h, self.epsbar_d)
        # if self.h_hatprime<0:
            # time.sleep(5)

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
