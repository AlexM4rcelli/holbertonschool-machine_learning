#!/usr/bin/env python3

import numpy as np

class Node:
	def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
		self.feature = feature
		self.threshold = threshold
		self.left_child = left_child
		self.right_child = right_child
		self.is_leaf = False
		self.is_root = is_root
		self.sub_population = None
		self.depth = depth
		self.lower = {}
		self.upper = {}


	def max_depth_below(self):
		max_depth = [0, 0]


		def calculate_depth(node, depth, side):
			if node is None:
				max_depth[side] = max(max_depth[side], depth)
			else:
				calculate_depth(node.left_child, depth + 1, 0)
				calculate_depth(node.right_child, depth + 1, 1)

		calculate_depth(self.left_child, 0, 0)
		calculate_depth(self.right_child, 0, 1)

		return max(max_depth)


	def count_nodes_below(self, only_leaves=False):
		if self.is_leaf:
				return 1
		if only_leaves:
			return self.left_child.\
				count_nodes_below(only_leaves=True) + self.right_child.\
				count_nodes_below(only_leaves=True)

		return 1 + self.left_child.\
			count_nodes_below(only_leaves=False) + self.right_child.\
			count_nodes_below(only_leaves=False)
	
	
	def left_child_add_prefix(self,text):
		lines=text.split("\n")
		new_text="    +--"+lines[0]+"\n"
		for x in lines[1:] :
			new_text+=("    |  "+x)+"\n"
		return (new_text)


	def right_child_add_prefix(self, text):
		lines = text.split("\n")
		new_text = "    +--"+lines[0]+"\n"
		for x in lines[1:]:
			if x:
				new_text += ("       " + x) + "\n"
		return new_text


	def __str__(self):
		node_str = ""
		left_str = ""
		right_str = ""

		if not self.is_root:
			node_str += f"-> node "
		else:
			node_str += f"root "

		node_str += f"[feature={self.feature}, threshold={self.threshold}]"

		if self.left_child:
			left_str += self.left_child_add_prefix(
	   			self.left_child.__str__().strip())
		if self.right_child:
			left_str += self.right_child_add_prefix(
	   			self.right_child.__str__().strip())

		return f"{node_str}\n{left_str}{right_str}"


	def get_leaves_below(self):
		if self.is_leaf:
			return [self]
		else:
			left_below = self.left_child.get_leaves_below()
			right_below = self.right_child.get_leaves_below()
			return left_below + right_below


	def update_bounds_below(self):
		if self.is_root: 
			self.upper = { 0:np.inf }
			self.lower = {0 : -1*np.inf }

		for child in [self.left_child, self.right_child]:
			child.upper = self.upper.copy()
			child.lower = self.lower.copy()
			if child == self.left_child:
				child.lower[self.feature] = self.threshold

			elif child == self.right_child:
				child.upper[self.feature] = self.threshold
				
   

		for child in [self.left_child, self.right_child]:
			child.update_bounds_below()

class Leaf(Node):
	def __init__(self, value, depth=None):
		super().__init__()
		self.value = value
		self.is_leaf = True
		self.depth = depth


	def max_depth_below(self) :
		return self.depth


	def count_nodes_below(self, only_leaves=False) :
		return 1
	
	
	def __str__(self):
		return (f"-> leaf [value={self.value}] ")


	def get_leaves_below(self):
		return [self]


	def update_bounds_below(self) :
		pass


class Decision_Tree():
	def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
		self.rng = np.random.default_rng(seed)
		if root:
				self.root = root
		else:
				self.root = Node(is_root=True)
		self.explanatory = None
		self.target = None
		self.max_depth = max_depth
		self.min_pop = min_pop
		self.split_criterion = split_criterion
		self.predict = None

	def depth(self) :
		return self.root.max_depth_below()
	
	
	def count_nodes(self, only_leaves=False) :
		return self.root.count_nodes_below(only_leaves=only_leaves)
	
	
	def __str__(self):
		return self.root.__str__()  


	def get_leaves(self) :
		return self.root.get_leaves_below()


	def update_bounds(self) :
		self.root.update_bounds_below()
