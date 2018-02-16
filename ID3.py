import pandas as pd
from math import log
from random import randint
from collections import Counter
import sys

class decisionNode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col # column index of criteria being tested
        self.value=value # value necessary to get a true result
        self.results=results # dict of results for a branch, None for everything except endpoints
        self.tb=tb # true decision nodes 
        self.fb=fb # false decision nodes
        self.leaf = False # tells whether a node is a leaf node
        self.depth = 0 # depth data helps to print the tree easily
        self.sample_max = None # maximum sample size to help pruning
    def assignAttributes(self,col=-1,value=None,results=None,tb=None,fb=None,lf=False,dp=0,max_sample = None):
        self.col=col # column index of criteria being tested
        self.value=value # vlaue necessary to get a true result
        self.results=results # dict of results for a branch, None for everything except endpoints
        self.tb=tb # true decision nodes 
        self.fb=fb # false decision nodes
        self.leaf = lf
        self.depth = dp
        self.sample_max = max_sample
	
# Entropy calculation (using both Information Gain and Variance Impurity Heuristics(returnVal = 'VI'))
# returns the entropy for each unique value in the column and the overall entropy
def get_entropy(df,columns,returnVal='IG'):
	# df -- dataframe object to work upon
	# columns -- columns with first column representing target variable folowed by attributes
	log2=lambda x:log(x)/log(2) # get the log base 2 function from log base 10
	tem = {0 : 'negative' , 1 : 'positive'} # object used to map 0 to negative and 1 to positive values
	tem_rev = {'negative' : 0 , 'positive' : 1} # object used to remap negative to 0 and so on
	t = {}
	data_0 = df.loc[df[columns[0]] == 0 ,columns] # get all the column data in the 'columns' array with negative(0) target variable
	data_1 = df.loc[df[columns[0]] == 1 ,columns] # get all the column data in the 'columns' array with positive(1) target variable
	data_counts_0 = get_unique_data_counts(data_0) # get the data count for unique values in each colum in columns
	data_counts_1 = get_unique_data_counts(data_1) # get the data count for unique values in each colum in columns
	data_counts = [data_counts_0 , data_counts_1] # make an array to iterate both
	results = {}
	column_Entr = {} 
	total_Entropy = {}
	# Initialize the results object for each column
	for column in columns:
		results[column] = {'negative' : [] , 'positive': []}
		column_Entr[column] = []
	# prepare the array for negative and positive incident counts in the same object . e.g. {'XB' : {0 : [10,20] , 1 : [20,30]}}
	for data in data_counts:
		for obj in data.items():
			total = 0.00
			for values in obj[1].items():
				total = total + values[1]
			column_Entr[obj[0]].append(total)
	# for type Information Gain
	if returnVal == 'IG':
		entropies = {}
		try:
			for values in column_Entr:
				entropy = 0.00
				for value in column_Entr[values]:
					p=float(value)/sum(column_Entr[values]) 
					entropy=entropy-p*log2(p)
				column_Entr[values]=entropy
			for data in data_counts:
				for obj in data.items():
					for d in obj[1].items():
						results[obj[0]][tem[d[0]]].append(d[1])
			for r in results.items():
				t = {}
				for data in results[r[0]].items():
					#print('sum : ', sum(data[1]) )
					entropy = 0
					for value in data[1]:
						p=float(value)/sum(data[1]) 
						entropy=entropy-p*log2(p)
						t[tem_rev[data[0]]] = entropy
					t['overall'] = column_Entr[r[0]]
					entropies[r[0]]=t    
		except:
			pass
		return entropies 
	# For Variance Impurity Heuristic
	elif returnVal == 'VI':
		impurities = {}
		try:
			for values in column_Entr:
				impurity = 1.00
				for value in column_Entr[values]:
					impurity= impurity*float(value)/sum(column_Entr[values]) 
					impurities[values]=impurity    
		except:
			pass
		return impurities
# Calculate Information Gain on a specific column 
def get_InformationGain(df, S , A , type_R= 'IG'):
	# S -- Target Variable
	# A -- Attribute for which Information Gain is needed
	# df -- dataframe object to work upon
	columns = [S,A]
	summation = 0.00
	entropy_s = 0
	if not df.empty:
		entropies = get_entropy(df, columns,type_R)
		if len(entropies) >0:
			if type_R == 'IG' :
				entropy_s = entropies[S]['overall']
			else : 
				entropy_s = entropies[S]
			count_s = df[S].count()
			keys = df[A].unique()
			counts=df[A].value_counts()
			data_counts = dict(zip(keys, counts))
			try:
				for r in data_counts.keys():
					p=float(data_counts[r])/sum(counts) 
					if type_R == 'IG' :
						summation=float(summation)+float(p*entropies[A][r])
					else:
						summation=float(summation)+float(p*entropies[A])
			except:
				pass
		return float(entropy_s-summation) if type_R == 'IG' else entropy_s-summation
	else:
		return 'error : empty dataframe'
# get unique data count in each column of the dataframe
def get_unique_data_counts(data):
	# data -- dataframe object
	try:
		result = {}
		if not isinstance(data , pd.Series):
			for column in data:
				result[column] = __uniqueCounts(data,column)
		else:
			result[data.name] = dict(zip(list(set(data)), pd.Series(Counter( data ))))
		return result
	except:
		print('column' , ' : ' , column)
		print(data.columns)
# Get unique data count in a single column of dataframe
def __uniqueCounts(data_frame, column):
	return data_frame.groupby(column).size().to_dict()

# partitioning function based on value
def __getSubset(df,column, value):
	set1 = df[df[column] == value] #Observations equal to value are in set 1
	set2 = df[df[column] != value] #Observations not equal to value are in set2 
	return (set1, set2)

# the ID3 algorithm as described in Tom Mitchell
def ID3(training_data , target_attribute,root,depth,heuristic='IG',max_dep=[]):
	# training_data -- dataframe object to work upon
	# target_attribute -- the target column name
	# root -- the decisionNode root object
	# heuristic -- default 'IG' - Information Gain -- use 'VI' for Variance Impurity
	if training_data.empty :
		return 
	else:
		classes = __uniqueCounts(training_data,target_attribute)
		attributes = list(training_data.columns)[:-1]
		if len(classes) == 1:
			root.results = list(classes.keys())[0]
			root.depth = depth-1
			max_dep.append(depth-1)
			root.col = None
			root.leaf=True
			return
		elif len(attributes) == 0:
			root.results = max(classes, key=classes.get)
			root.depth = depth-1
			max_dep.append(depth-1)
			root.col = None
			root.leaf = True
			return
		else :
			max_gain = 0
			best_attribute = None
			for attribute in attributes:
				temp_IG = get_InformationGain(training_data,target_attribute,attribute,heuristic)
				if temp_IG>=max_gain:
					max_gain = temp_IG
					best_attribute = attribute		
			root.col=best_attribute
			root.depth = depth
			root.fb = decisionNode()
			root.tb = decisionNode()
			root.fb.assignAttributes(value=0,max_sample = max(classes, key=classes.get))
			root.tb.assignAttributes(value=1)
			unique_values = sorted(training_data[best_attribute].unique())
			(set1 , set2) = __getSubset(training_data,best_attribute,unique_values[0])
			ID3(set1.drop(best_attribute,axis=1),target_attribute,root.fb,depth+1)
			ID3(set2.drop(best_attribute,axis=1),target_attribute,root.tb,depth+1)
	return (root,max_dep)

# Print sign 
def printTab(n):
	#n - number of times to print 
	for i in range(0,n):
		print('|--->',end="")
		
# Print tree
def printTree(root):
# root of the decicion tree 
	if root.depth == 1:
		print(root.col , '= ' , end="")
	else :
		if not root.leaf:
			print(root.value , ': \n' )
			printTab(root.depth)
			print(root.col ,' = ' , end="")
		elif root.col is None:
			print(root.value , ' : ' , root.results ,' \n ' , end="\n")
	if root.fb is not None:
		printTree(root.fb)
	if root.depth == 1:
		print(root.col , '= ' , end="")
	else :
		if not root.leaf :
			printTab(root.depth)
			print(root.col ,' = ' , end="")
	if root.tb is not None:
		printTree(root.tb)

# predict a given row of a new dataframe
def predict(test_set , trained_Dtree):
	# test_set  -- a row in test dataframe
	# trained_Dtree --  the trained Decision tree model
	if trained_Dtree.col is None:
		return trained_Dtree.results
	elif trained_Dtree.leaf :
		return trained_Dtree.results
	if test_set[trained_Dtree.col] == 0 :
		return predict(test_set,trained_Dtree.fb)
	else:
		return predict(test_set,trained_Dtree.tb)
# get the accuracy of the trained model
def accuracy(test_data , trained_model):
	# test_set  -- a row in test dataframe
	# trained_model --  the trained Decision tree model
	count = 0
	for num in range(0,len(test_data)):
		if predict (test_data.loc[num],trained_model) == test_data.loc[num]['Class']:
			count = count+1
	return (count/len(test_data))*100

# function to replace a subtree by a given leaf node
def replace_node_by_leaf(root,node):
	# root -- root of the tree (Decision Tree)
	# node --  the node to be replaced
	leaf=get_leaf_node(node)
	if node is None :
		return root
	if leaf is None :
		return root
	node.fb = leaf.fb
	node.tb = leaf.tb
	node.value = leaf.value
	node.results = node.sample_max
	node.depth = leaf.depth
	node.col = leaf.col
	return root

# returns the leaf node from a given node
def get_leaf_node(tree):
	rand_num = randint(0, 1)
	if tree is None:
		return tree
	if tree.leaf :
		return tree
	if rand_num == 0:
		return get_leaf_node(tree.fb)
	else:
		return get_leaf_node(tree.tb)
# returns a random node with a given maxDepth
def get_random_node(tree,maxDepth=2):
	rand_num = randint(0, 1)
	if tree.tb is None or tree.fb is None:
		return tree
	if tree.depth>=maxDepth or tree.tb.leaf or tree.fb.leaf :
		return tree
	if rand_num == 0:
		return get_random_node(tree.fb)
	else:
		return get_random_node(tree.tb)
# copy the tree and return the root of the new tree
def copy(tree):
	if tree is None:
		return
	else :
		temp = decisionNode()
		temp.col=tree.col # column index of criteria being tested
		temp.value=tree.value # vlaue necessary to get a true result
		temp.results=tree.results # dict of results for a branch, None for everything except endpoints
		temp.leaf = tree.leaf
		temp.depth =tree.depth
		temp.sample_max = tree.sample_max
		temp.tb=copy(tree.tb) # true decision nodes 
		temp.fb=copy(tree.fb) # false decision nodes
	return temp
# The post pruning function 
def post_pruning(tree,L,K,validation_set):
	D_best = tree
	acc_b = accuracy(validation_set,D_best)
	for i in range(1,L):
		D_prime = copy(D_best)
		M = randint(1, K)
		for j in range(1,M):
			node_rand = get_random_node(D_prime,randint(1,15))
			if node_rand is None: 
				break
			D_prime=replace_node_by_leaf(D_prime,node_rand)
		t=accuracy(validation_set,D_prime)
		if t > acc_b :
			D_best = D_prime
	return D_best
def main():
	args = (sys.argv)
	L = int(args[1])
	K = int(args[2])
	training_set = str(args[3])
	testing_set = str(args[5])
	validation_set = str(args[4])
	to_print = str(args[6])
	
	df1 = pd.read_csv(training_set)
	df2 = pd.read_csv(testing_set)
	df3 = pd.read_csv(validation_set)
	
	(result,depth)= ID3(df1,'Class',decisionNode(),1)
	(result_VI,depth_VI)= ID3(df1,'Class',decisionNode(),1,'VI')
	
	print('Accuracy with Information Gain Heuristic : ' , accuracy(df2,result) ,' %')
	print('Accuracy with Variance Impurity Heuristic : ' , accuracy(df2,result) ,' %')

	post_tree = post_pruning(result,L,K,df2)

	print('Accuracy of Pruned Tree : ' , accuracy(df3,post_tree) ,' %')
	
	if to_print == 'yes':
		print('Decision Tree with Information Gain heuristic')
		printTree(result)
		print('Decision Tree with Variance Impurity heuristic')
		printTree(result_VI)
	
if __name__ == "__main__":
	main()
