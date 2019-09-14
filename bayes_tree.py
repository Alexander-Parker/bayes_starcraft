import pandas as pd
import numpy as np
from anytree import Node, RenderTree, PreOrderIter
from scipy.stats import norm
import copy
import sys
import json


class tree():

    def __init__(self):
        """
        A class for memorizing build order data and making build predictions based on game observations
        ---
        Inputs:
        """
        self.read_root = Node('root',mean=0,freq=1)
        self._is_fit = False
        self._is_copied = False
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

        last = self.read_root #Begin at the root node and go down

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
        # We dont want to fit if not all builds are in the tree yet

        for node in PreOrderIter(self.read_root):
            try:
                node.mean = np.mean(node.obs)
                if len(node.obs) <= cutoff:
                    node.std = default
                else:
                    node.std = np.std(node.obs)
                del node.obs
            except AttributeError:
                assert node is self.read_root, "Node other than root is missing observations"
        self._is_fit = True
        return
        
    def prepare(self):
        """
        Makes a copy of the memorized tree to adjust frequencies on
        TBU: MAKE A MORE EFFICIENT COPIER
        """
        assert self._is_fit, "Fit model before making the prediction tree"
        sys.setrecursionlimit(100000)
        self.write_root = copy.deepcopy(self.read_root)
        sys.setrecursionlimit(1000)
        self._is_copied = True
        return  

    def update_tree(self,obs,prune_sd=3):
        assert self._is_copied, "Tree must be copied before updating"

        def get_nodes(obs_key,obs_time,root):
            if len(root.children) == 0:
                zero_list.append(root)

            for node in root.children:
                if node.freq <= 0:
                    continue
                if node.key == obs_key:
                    bayes_list.append(node)
                    continue
                if (node.mean - prune_sd * node.std) > obs_time:
                    zero_list.append(node)
                    continue

                get_nodes(obs_key,obs_time,node)
            return

        def update_parents(delta,node):
            node.freq = node.freq + delta
            if node.parent:
                return update_parents(delta,node.parent)
            else:
                return

        def update_children(old_freq,new_freq,node):
            children = node.children
            if len(children) == 0:
                return
            for node in node.children:
                node.freq = node.freq / old_freq * new_freq
                update_children(old_freq,new_freq,node)
            return
    
        bayes_list = []
        zero_list = []
        
        get_nodes(obs[0],obs[1],self.write_root)
        
        #Freq Check
        freq = 0
        for node in bayes_list:
            freq += node.freq
        for node in zero_list:
            freq += node.freq
        print(f'Freq Check: {freq}')
        
        bayes_list_cdf = [norm.cdf(obs[1], node.mean, node.std) for node in bayes_list]
        zero_list_cdf = [0 for _ in zero_list]
        
        all_nodes = bayes_list + zero_list
        all_cdfs = bayes_list_cdf + zero_list_cdf
        
        
        denom = 0
        numer = []
        for index,node in enumerate(all_nodes):
            val = node.freq * all_cdfs[index]
            numer.append(val)
            denom += val
            
        node_prob = []
        for num in numer:
            node_prob.append(num/denom)
            
        for index,node in enumerate(all_nodes):
            old = node.freq
            new = node_prob[index]
            update_parents(new-old,node)
            update_children(old,new,node)
            
        
        return

    def predict(self,time,attribute='key',read_root=False):
        assert self._is_copied, "Tree must be copied before prediction"

        predict_parents = []
        predict_children = []

        def get_nodes(time,root):
            if root.mean >= time:
                predict_parents.append(root)
                predict_children.append(list(root.children))
                return 

            for node in root.children:
                if node.freq <= 0:
                    continue

                get_nodes(time,node)

            return

        if read_root:
            get_nodes(time,self.read_root)
        else:
            get_nodes(time,self.write_root)

        weighted_predictions = []

        for index,parent in enumerate(predict_parents):
            child_freq = norm.cdf(time,parent.mean,parent.std)
            parent_freq = 1 - child_freq
            weighted_predictions.append((getattr(parent,attribute), parent_freq * parent.freq ))
            for child in predict_children[index]:
                weighted_predictions.append((getattr(child,attribute), child_freq * child.freq ))

        return weighted_predictions

    def export_json(self,fp=None,read_tree=False):
        node_list = []
        next_id = 0
        
        def node_linker(root_node,parent_id=None):
            nonlocal next_id
            id_num = next_id
            next_id += 1
            if parent_id is not None:
                node_list.append({
                    "id": id_num,
                    "name": root_node.name,
                    "parent": parent_id,
                    "freq": root_node.freq
                })
            else:
                node_list.append({
                    "id": id_num,
                    "name": 'root',
                    "freq": root_node.freq
                })
            for child in root_node.children:
                node_linker(child,id_num)
            return
        
        if read_tree:
            node_linker(self.read_root)
        else:
            node_linker(self.write_root)
        
        if fp:
            with open(fp, 'w') as fout:
                json.dump(node_list, fout)
            return None
        else:
            return json.dumps(node_list)