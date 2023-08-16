#   DecisionTree.py
#
#   Basic implementation of a decision tree for binary
#   classification problems
#   Written by Jeff Long for CMPT 317, University of Saskatchewan


import math as math

class Decision_Treenode(object):

    def __init__(self):
        return
        
    def classify(self, sample):
        """ returns the label for the given sample """

        
class Label_Node(Decision_Treenode):

    def __init__(self, label):
        self.label = label
        
    def classify(self, sample):
        return self.label
        
    def __str__(self):
        if self.label != None:
            return "Label " + self.label
        else:
            return "Label None"
        
        
class Feature_Node(Decision_Treenode):

    def __init__(self, feature, values):
        """ returns a new feature node that splits on the given feature
        
        :params:
        feature: string. name of the categorical feature this node will split on
        values: list of all possible values for that feature
        
        returns: a new feature node with no children yet
        """
        self.feature = feature
        self.children = {}
        for v in values:
            self.children[v] = None
            
            
    def classify(self, sample):
        sample_value = sample[self.feature]
        child_node = self.children[sample_value]
        return child_node.classify(sample)
            

    def __str__(self):
        return "Feature " + self.feature
               
        
class Decision_Tree(object):

    def __init__(self):
        self.root = None
        self.outputname = ""
        
        
    def build(self, filename):
        """ builds a binary decision tree from data in the given file name
        file must be plain text with one column for each feature
        It is assumed the first column is just the sample ID and the
        last column is the label. columns are white-space separated, 
        and sample labels must all values 'yes' or 'no'
        
        :params:
        filename: string. name of the file to open
        """
        f = open(filename, 'r')
        features = f.readline()
        features = features.strip().split()
        
        # get rid of the sample_id
        features = features[1:]
        
        # get rid of the label name in the last column
        self.outputname = features[-1]
        features = features[:-1]
        
        # construct a list of all possible values for each feature,
        # so that we're not limited to binary features
        feature_vals = {}
        for feat in features:
            feature_vals[feat] = []
        
        data = []
        for line in f:
            line = line.strip().split()
            line = line[1:] # get rid of the sample_id
            s = {}
            s["label"] = line[-1]
            for i in range(len(features)):
                s[features[i]] = line[i]
                if line[i] not in feature_vals[features[i]]:
                    feature_vals[features[i]].append(line[i])
                
            data.append(s)
            
        
        f.close()
        
        # store the possible values for features for later queries if needed
        self.features = feature_vals
        majority = self.get_majority_label(data)
        
        if len(data) > 0:
            self.root = self.__build_rec(data, feature_vals, majority)
            
##############################################################################
            

    def get_best_feature(self, data, features):
        """ returns the name of the feature with the highest information gain
        to use for a split, given the current data points
        
        :parms:
        data: a list of records representing the data.  Each record MUST have a field called "label", and other fields must be the name of a feature
        features: dictionary mapping a feature name to a list of all possible values for that feature
        
        returns: the name (string) of the most informative feature on which to split the given data
        """
        #TODO: currently this is identical to "get_some_feature()".  Improved it by measuring the information gain
        # of each feature!

        # Define a function to compute the entropy of items
        def entropy(items):
            # Initialize variables
            counts = dict()
            total = 0

            # Loop through each item
            for item in items:
                label = item['label']
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1
                total += 1

            return sum(-(counts[label] / total) * math.log(counts[label] / total) for label in counts)

        # Define a function to split items based on a feature name
        def split(items, feature):
            # Initialize the variables
            splits = dict()
            total = 0

            # Check each value in the features
            for value in features[feature]:
                splits[value] = [item for item in items if item[feature] == value]
                total += len(splits[value])

            return splits, total

        # Compute Entropy(T)
        E_T = entropy(data)

        # Compute the information gains for each feature
        IG = dict()
        for feature in features:
            splits, total = split(data, feature)
            Sum_E_Tv = 0

            # Compute the total entropy
            for value in splits:
                Sum_E_Tv += len(splits[value]) / total * entropy(splits[value])

            # Compute information gain
            # IG = Entropy(T) - Sum (T_v / T * Entropy(T_v))
            IG[feature] = E_T - Sum_E_Tv

        # Get the feature with the most information gain
        return max(IG)

    def get_some_feature(self, data, features):
        """ placeholder to compare with getting the best feature
        """
        return list(features.keys())[0]


###############################################################################
     
    def get_majority_label(self, data):
        """ returns the majority label from the data set.  if there is a tie,
        then an arbitrary label is returned """
        
        histo = {}
        result = ""
        for d in data:
            label = d["label"]
            result = label
            if label not in histo:
                histo[label] = 1
            else:
                histo[label] += 1
        
        for label in histo:
            if histo[label] > histo[result]:
                result = label
        return result
       
     
    def __build_rec(self, data, features, mostcommon):
        """ recursively builds a decision tree with the given data and features,
        returning the root of that tree
        
        :params:
        data: a list of records representing the data.  Each record MUST have a field called "label", and other fields must be the name of a feature
        features: dictionary mapping a feature name to a list of all possible values for that feature
        mostcommon: string. most common label of the parent node.  This is needed in the event
        we find a feature combination of which we have seen no examples
        """
        # base case 1: data is empty, meaning we don't have any point with the current feature values
        # use the most common label from the parent's data set
        if len(data) == 0:
            return Label_Node(mostcommon)
        # base case 2: features are empty, so return label node with majority class
        elif len(features.keys()) == 0:
            majority = self.get_majority_label(data)
            return Label_Node(majority)
        else:
        
            # base case 3: all the remaining data has the same label, so features don't matter
            all_same = True
            first = data[0]["label"]
            for d in data:
                if d["label"] != first:
                    all_same = False
                    break
                    
            if all_same:
                return Label_Node(first)
            else:
                # recursive case: split on some feature
            
                # get the majority label to pass down in case there are child nodes with no data
                majority = self.get_majority_label(data)
                
                # get the feature on which to split
                feat = self.get_best_feature(data, features)
                # feat = self.get_some_feature(data, features)
                
                # temporarily remove this feature for the recursive calls
                feat_vals = features.pop(feat)
                node = Feature_Node(feat, feat_vals)
                
                # for each possible value of the chosen feature, partition the data based on that value
                for v in feat_vals:
                    partition = []
                    for d in data:
                        if d[feat] == v:
                            partition.append(d)
                            
                    child = self.__build_rec(partition, features, majority)
                    node.children[v] = child
                            
                # make sure to add the feature back!
                features[feat] = feat_vals
                return node
    

    def classify(self, sample):
        """ returns the classification label of the given sample
        
        :params:
        sample: a record represeting a data point.  keys of the record must be
        feature names that are in the tree
        
        returns: the label (string) for the given sample """
        return self.root.classify(sample)
        
    def size(self):
        """ returns the total number of nodes of any kind
        in this decision tree """
        return self.__size_rec(self.root)
        
    def __size_rec(self, node):
        if node == None:
            return 0
        elif isinstance(node, Label_Node):
            return 1
        else:
            numchildren = 0
            for child in node.children.values():
                numchildren += self.__size_rec(child)
                
            return 1 + numchildren
    
    
    def __str__(self):
        result = "Decision tree to determine the " + self.outputname + " of a data point.\n******\n"
        # do a breadth-first traversal of the tree to print. not a perfect visualization, but okayish
        queue = []
        # keep track of the depth of each node as we put it into the queue
        t = ("Root", self.root, 0)
        queue.append(t)
        cur_depth = 0
        layer = []
        
        while len(queue) > 0:
            cur = queue.pop(0)
            parent_value = cur[0]
            node = cur[1]
            depth = cur[2]
            if depth != cur_depth:
                cur_depth += 1
                result += " |--| ".join(layer)
                result += "\n\nNodes at layer " + str(cur_depth) + ": "
                layer = []
            layer.append(parent_value + " " + str(node))
            if isinstance(node, Feature_Node):
                for v, c in node.children.items():
                    next_node = (node.feature + "=" + v, c, cur_depth+1)
                    queue.append(next_node)
            
        result += " |--| ".join(layer)    
        return result
        
        