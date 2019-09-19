import pandas as pd
import numpy as np
from anytree import Node, RenderTree, PreOrderIter
from scipy.stats import norm
import copy
import sys
import json
import math


class tree():

    def __init__(self):
        """
        A class for memorizing build order data and making build predictions based on game observations
        ---
        Inputs:
        """
        self.root = Node('root',mean=0,freq=1)
        self.hist_freqs = []
        self._is_fit = False
        return 
        
    def add_build(self,build):
        """
        Function designed to update tree with a specific build order, adding nodes where needed
        and updating node values in both existing and created nodes 
        ---
        Inputs:
        build: dictionary that must contain 'build', 'freq' and 'time' keys with an optional 'name' key.
            Any values of additional keys will be added as attributes to each node travelled over.
            Key-Value Descriptions:
            build: List of tuples, with each tuple representing (unit_ID, count of unit_ID in that game)
            time: list of times (in seconds) corresponding to build events
            freq: float representing the occurance of this build 
            name: optional string to represent the name of a node
        """
        #Check that we have not fit the nodes for mean and std yet
        assert not self._is_fit, 'Cannot add builds to a fit tree'

        #Check that we have the right keys
        required_keys = ['build','freq','time']
        optional_keys = ['name'] #Other optional keys can be specified, but these have a specific behavior
        assert all(key in build.keys() for key in required_keys), 'Dictionary requires buid,freq and time keys'

        last = self.root #Begin at the root node and go down

        for index,event in enumerate(build['build']):
            change_flag = 0 

            #look to see if the child node already exists for our event
            for child in last.children: 
                if event == child.key:
                    new = child
                    change_flag = 1
                    break
            
            #if not, make one
            if change_flag == 0:
                try:
                    name = build['name'][index]
                except KeyError:
                    name = event

                new = Node(name,key=event,obs=[],freq=0,mean=0,std=0)
                new.parent = last

            #add these to new and old nodes
            new.obs.append(build['time'][index])
            new.freq += build['freq']


            #add custom keys, whose value is assumed to be a dictionary (used for cluster freq right now)
            for key in list(filter(lambda key: key not in (required_keys + optional_keys),build.keys())):
                try:
                    attr = getattr(new,key)
                    if build[key] in attr.keys():
                        attr[build[key]] += 1
                    else:
                        attr[build[key]] = 1
                except:
                    attr = setattr(new,key,{build[key]: 1})
                


            #set new parent to look at next event 
            last = new 
            
        return

    def fit_nodes(self,cutoff,default):
        """
        Converts node time observations into a mean and standard deviation, to be used in probability calculations.
        Once this is done, no builds may be added. A cutoff and default standard deviation must be specified to
        handle cases of few observations (occurs at greater depth).
        ---
        Inputs:
        cutoff: Integer. Observation counts at or below the cutoff will use the default rather than calculated standard deviation
        default: Float/Integer. Default standard deviation to be used in cases of few observations.
        """
        freq_list = []
        for node in PreOrderIter(self.root):
            freq_list.append(node.freq)
            try:
                node.mean = np.mean(node.obs)
                if len(node.obs) <= cutoff:
                    node.std = default
                else:
                    node.std = np.std(node.obs)
                del node.obs
            except AttributeError:
                assert node is self.root, "Node other than root is missing observations"

        self.hist_freqs = freq_list #allows us to go back to memorized freqs without having to relearn the tree
        self._is_fit = True
        return
        
    def reset(self):
        """
        Resets tree back to hist freqs
        """
        assert self._is_fit, "Fit model before resetting tree"
        for index,node in enumerate(PreOrderIter(self.root)):
            node.freq = self.hist_freqs[index]
        return  

    def update_tree(self,obs,prune_sd=5):
        """
        Updates tree frequencies for a given observation
        ---
        Inputs:
        obs: Tuple containning observation key and observation time. Observation key is a tuple of unit_type_id and unit type count
            Time is an integer representing seconds elapsed in the game.
        prune_sd: Updater will stop looking for a matching key node if the current nodes time is prune_sd sigma above the observation time.
            Heurisitic so the entire tree doesnt have to be searched for earlier observations.
        """
        assert self._is_fit, "Tree must be fit before updating"

        # function to force down extremely low probabilities
        def round_down(num):
            return num * (num > 9.9999999999e-14)

        def get_nodes(obs_key,obs_time,root):
            """
            Identifies nodes to evaluate with Bayes theorem. The idea is that we need a collection of nodes whose frequencies sum to one
            ---
            Inputs:
            obs_key: Observation key is a tuple of unit_type_id and unit type count
            obs_time: Time is an integer representing seconds elapsed in the game.
            root: parent node to begin search from (this will change during recursion)
            """
            # if parent doesnt have any children, then we'll need it to keep our freqs summing to 1
            if len(root.children) == 0:
                zero_list.append(root)

            for node in root.children:
                if node.freq <= 0: # we dont care about things that cant happen
                    continue
                if node.key == obs_key: # these nodes are eligible for bayes since they contain the correct observation
                    bayes_list.append(node) 
                    continue
                if (node.mean - prune_sd * node.std) > obs_time: #heurisitic to stop wasteful recursion 
                    zero_list.append(node)
                    continue

                get_nodes(obs_key,obs_time,node) # if there are remaining children, we need to keep looking down branch paths
            return

        def update_parents(delta,node):
            # add change in probability to all nodes "upstream" of evaluated node
            node.freq = round_down(max([node.freq + delta,0])) 
            if node.parent:
                return update_parents(delta,node.parent)
            else:
                return

        def update_children(old_freq,new_freq,node):
            # children frequencies should maintain the same ratio to the evaluated parent frequency 
            children = node.children
            for child in children:
                if node.freq <= 0:
                    child.freq = 0
                else:
                    child.freq = round_down(min([child.freq / old_freq * new_freq,node.freq]))
                update_children(old_freq,new_freq,child)
            return

        def calc_cdf(x,mu,sigma):
            # sigma is 0 for starter units since we always instantly have them - this breaks the norm.cdf function
            if sigma == 0:
                return 1 *(x>=mu)
            else:
                return norm.cdf(x, mu, sigma)
    
        bayes_list = []
        zero_list = []
        
        get_nodes(obs[0],obs[1],self.root)
        
        # Freq Check: This should always be very,very close to 1 since the idea is we are grabbing a full range of possibilities at 
        # different timings / build sequence depths
        
        # We end up ratio-ing by this freq to make sure that rounding errors dont compund into larger errors over multiple updates
        freq = 0
        for node in bayes_list:
            freq += node.freq
        for node in zero_list:
            freq += node.freq
        print(f'Freq Check: {freq}')
        
        bayes_list_cdf = [calc_cdf(obs[1], node.mean, node.std) for node in bayes_list]
        zero_list_cdf = [0 for _ in zero_list]
        
        all_nodes = bayes_list + zero_list
        all_cdfs = bayes_list_cdf + zero_list_cdf
        
        
        denom = 0
        numer = []
        for index,node in enumerate(all_nodes):
            val = node.freq * all_cdfs[index]
            numer.append(val)
            denom += val

        if denom <= 0: #this occurs when we are predicting on a build not in our data set. We will just ignore obs that break our model
            return None
            
        node_prob = []
        likelihoods = []
        for index,num in enumerate(numer):
            node_num = num/denom * (1/freq)
            node_num = max([0,node_num])
            node_prob.append(node_num)
            likelihoods.append(num/denom/all_nodes[index].freq) 
            
        for index,node in enumerate(all_nodes):
            old = float(node.freq)
            new = float(node_prob[index])
            update_parents(new-old,node.parent)
            node.freq = round_down(new)
            update_children(old,new,node)
        return likelihoods #You could make a SSE-like metric with this to assess observation importance 

    def predict(self,time,attribute='key'):
        """
        Returns the attributes of a list of nodes occuring after a specified time with the respective node frequencies. 
        This allows for predicting attributes such as next build order step or strategy cluster.
        ---
        Inputs:
        time: Integer in seconds that represents elapsed game time
        attribute: String represent name of attribute to return predictions for 
        """
        assert self._is_fit, "Tree must be fit before prediction"

        predict_list = []

        def get_nodes(time,root):
            if root.mean >= time:
                predict_list.append(root)
                return 

            children = root.children

            # if there are no more children, just return the latest node
            # not a great assumption, but keeps frequencies adding to 1
            if len(children) == 0:
                predict_list.append(root)
                return

            for node in children:
                if node.freq <= 0:
                    continue

                get_nodes(time,node)

            return

        get_nodes(time,self.root)

        weighted_predictions = []

        for node in predict_list:
            weighted_predictions.append((getattr(node,attribute), node.freq ))

        return weighted_predictions

    def export_json(self,fp=None,root_cluster=13):
        """
        Exports tree as a JSON dictionary to an optional file path, otherwise returns as a string. Designed for Vega plotting.
        ---
        Inputs:
        fp: File path to save JSON file to
        big: Bool. True means that the entire tree will be exported. False means the tree up until no more splits will be exported.
        """
        node_list = []
        next_id = 0

        def node_linker(root_node,parent_id=None,force_add=False,big=False):
            nonlocal next_id
            force_flag = False
            if parent_id is not None:
                id_num = parent_id
                if len(root_node.children) > 1 or big:
                    force_flag = True
                if len(root_node.children) > 1 or force_add or big:
                    id_num = next_id
                    next_id += 1
                    node_list.append({
                        "id": id_num,
                        "name": root_node.name,
                        "parent": parent_id,
                        "freq": root_node.freq,
                        'cluster' : int(max(root_node.cluster, key=root_node.cluster.get))  #returns most frequent cluster for each node
                    })
            else:
                id_num = next_id
                next_id += 1
                node_list.append({
                    "id": id_num,
                    "name": 'root',
                    "freq": root_node.freq,
                    'cluster' : root_cluster
                })
            for child in root_node.children:
                node_linker(child,id_num,force_add=force_flag)
            return

        node_linker(self.root)

        if fp:
            with open(fp, 'w') as fout:
                json.dump(node_list, fout)
            return None
        else:
            return json.dumps(node_list)