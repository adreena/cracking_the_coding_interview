# ## P8: First Common Ancestor
# 
# Design an algorithm and write code to find the first common ancestor of 2 nodes in a binary tree. Avoid storing additional nodes in a data structure. Note: This is not necessarily a binary search tree

# In[18]:


# find paths from root to each of the nodes, find the first differnece between the paths
class TNode:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None
    def find_path(self, root, node_key, path):
        if root is None:
            return False
        
        path.append(root.data)
        if root.data == node_key:
            return True
        else:            
            if (root.left is not None and self.find_path(root.left, node_key, path)) or                (root.right is not None and self.find_path(root.right, node_key, path)):
                return True
            else:
                path.pop()
                return False 
    def FCA(self, root,  node1, node2):
        path1 = []
        output1 =  btree.find_path(root, node1, path1)
        path2 = []
        output2 =  btree.find_path(root, node2, path2)
        if not (output1 and output1):
            return None
        else:
            i = 0
            while i < len(path1) and i < len(path2):
                if path1[i] != path2[i]:
                    if i == 0:
                            return None
                    return path[i-1]
                i+=1
            return None

btree = TNode(3)
btree.left = TNode(5)
btree.left.left = TNode(6)
btree.left.right = TNode(2)
btree.left.right.left = TNode(7)
btree.left.right.right = TNode(4)
btree.right = TNode(1)
btree.right.left = TNode(0)
btree.right.right = TNode(8)

print(btree.FCA(btree, 5, 8))


# ## P9: BST Sequences
# 
# A binary search tree was created by traversing through an array from left to right and inserting each element. Given a binary search tree with distinct elements, print all oissible arrays that could have lef to this tree

# In[ ]:


# 4 2 1 3 5 6


# ## P10: Check Subtree
# 
# T1 and T2 are 2 very large binary trees, with T1 much bigger than T2. Create an algorithm to determine T2 is a subtree of T1.
# 
# A tree T2 is a subtree of T1 if there exists a node n in T1 such that the subtree of n is identical to T2. That is, if you cut off the tree at node n the 2 trees should be identical

# In[24]:


class TNode:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None
        
def is_indentical(tree1, tree2):
    if tree1 is None and tree2 is None:
        return True
    if (tree1 is None) ^ (tree2 is None):
        return False
    return tree1.data == tree2.data and is_indentical(tree1.left, tree2.left) and is_indentical(tree1.right, tree2.right)

def check_subtree(big_tree, small_tree):
    if is_indentical(big_tree, small_tree):
        return True
    return (big_tree.left is not None and check_subtree(big_tree.left, small_tree)) or            (big_tree.right is not None and check_subtree(big_tree.right, small_tree))

big_tree = TNode(26)
big_tree.left = TNode(10)
big_tree.left.left = TNode(4)
big_tree.left.right = TNode(6)
big_tree.left.left.right = TNode(30)
big_tree.right = TNode(3)
big_tree.right.right = TNode(3)

small_tree = TNode(10)
small_tree.left = TNode(4)
small_tree.right = TNode(6)
small_tree.left.right = TNode(30)

check_subtree(big_tree, small_tree)
