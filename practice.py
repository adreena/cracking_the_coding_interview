#!/usr/bin/env python
# coding: utf-8

# In[88]:


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        
class Stack:
    def __init__(self):
        self.top = None
    
    def is_empty(self):
        if self.top is None:
            return True
        else:
            return False
        
    def push(self, value):
        new_node = Node(value)
        if self.is_empty():
            self.top = new_node
        else:
            new_node.next = self.top
            self.top = new_node
    
    def pop(self):
        if self.is_empty():
            return None
        else:
            data = self.top.value
            self.top = self.top.next
            return data
        
    def peek(self):
        if self.is_empty():
            return None
        else:
            return self.top.value
        
mystack = Stack()
mystack.push(1)
mystack.push(2)
mystack.push(3)
while not mystack.is_empty():
    print(mystack.pop())


# In[4]:


class Queue:
    def __init__(self):
        self.first = None
        self.last = None
        
    def is_empty(self):
        if self.first is None:
            return True
        else:
            return False
    
    def add(self, value):
        if self.is_empty():
            self.first = Node(value)
            self.last = self.first
        else:
            self.last.next = Node(value)
            self.last = self.last.next
    
    def remove(self):
        if self.is_empty():
            return None
        else:
            data = self.first.value
            self.first = self.first.next
            if self.first is None:
                self.last = None
            return data

q = Queue()
q.add(1)
q.add(2)
q.add(3)

while not q.is_empty():
    print(q.remove())


# In[35]:



class Graph:
    def __init__(self, graph_elements):
        self.graph_elements = graph_elements
        
    def get_all_vertices(self):
        return [key for key,val in self.graph_elements.items()]

    def get_adjacent_vertices(self, vertex):
        all_verts = self.get_all_vertices()
        if vertex in all_verts:
            return self.graph_elements[vertex]
        else:
            return None
        
    def DFS(self, start, end):
        stack = Stack()
        stack.push((start, [start]))
        visited = [start]
        while not stack.is_empty():
            node, path = stack.pop()
            print(node, path)
            if node == end:
                return True, path
            adj_nodes = self.get_adjacent_vertices(node)
            for node in adj_nodes:
                if node not in visited:
                    visited.append(node)
                    stack.push((node, path+[node]))
        return False, None  
    
    def BFS(self, start, end):
        queue = Queue()
        queue.add((start, [start]))
        visited = [start]
        while not queue.is_empty():
            node, path = queue.remove()
            print(node,path)
            if node == end:
                return True, path
            adj_verts = self.get_adjacent_vertices(node)
            for vert in adj_verts:
                if vert not in visited:
                    visited.append(vert)
                    queue.add((vert, path+[vert]))
        return  False, None
_graph = {
    'a':['b', 'c'],
    'b':['d', 'e'],
    'c':['e'],
    'd':['f', 'e'],
    'e':['c', 'f'],
    'f':['e']
}
my_graph = Graph(_graph) 
# print(my_graph.get_all_vertices())
# print(my_graph.get_adjacent_vertices('c'))
print(my_graph.DFS('a','d'))
print(my_graph.BFS('a','d'))


# In[37]:


class BNode:
    def __init__(self, val):
        self.data = val
        self.right = None
        self.left = None

def depth_lists(root):
    lists = []
    current_list = [root]
    while len(current_list) > 0 :
        new_list = []
        lists.append(current_list)
        for item in current_list:
            if item.left is not None:
                new_list.append(item.left)
            if item.right is not None:
                new_list.append(item.right)
        current_list = new_list
        
    return lists
my_btree= BNode(6)
my_btree.left = BNode(4)
my_btree.right = BNode(8)
my_btree.left.left = BNode(3)
my_btree.left.right = BNode(5)
my_btree.right.right = BNode(10)
my_btree.right.left = BNode(7)

all_lists = depth_lists(my_btree)
for l in all_lists:
    for item in l:
        print(item.data)
    print('----')


# In[42]:


def get_depth(node):
    if node is None:
        return 0
    else:
        return max(get_depth(node.left)+1 , get_depth(node.right) + 1)

def check_banalced(root):
    if root is None:
        return True
    left_depth = get_depth(root.left)
    right_depth = get_depth(root.right)
    if left_depth != right_depth :
        return False
    if check_banalced(root.left) and check_banalced(root.right):
        return True
    return False
    
    
my_btree= BNode(6)
my_btree.left = BNode(4)
my_btree.right = BNode(8)
my_btree.left.left = BNode(3)
my_btree.left.right = BNode(5)
# my_btree.right.right = BNode(10)
# my_btree.right.left = BNode(7)
check_banalced(my_btree)


# In[45]:


class QueueStack:
    def __init__(self):
        self.old_stack = Stack()
        self.new_stack = Stack()
    
    def is_empty(self):
        if self.old_stack.is_empty() and self.new_stack.is_empty():
            return True
        else:
            return False
    
    def add(self, new_value):
        self.new_stack.push(new_value)
        
    def remove(self):
        if self.old_stack.is_empty():
            while not self.new_stack.is_empty():
                self.old_stack.push(self.new_stack.pop())
        return self.old_stack.pop()

    def peek(self):
        return self.new_stack.peek()
    
q = QueueStack()
q.add(1)
q.add(2)
q.remove()
q.add(3)
q.add(1)

while not q.is_empty():
    print(q.remove())


# In[59]:


class ThreeStackInOneArray:
    def __init__(self, size):
        self.first_idx = [0, int(size/3), int(2*size/3)]
        self.last_idx = [0, int(size/3), int(2*size/3)]
        self.array = [None] * size
        self.size = size
        
    def add(self, stack_id, value):
        if self.last_idx[stack_id] == self.first_idx[stack_id] + int(self.size/3) -1 :
            return 
        else:
            self.array[self.last_idx[stack_id]] = value
            self.last_idx[stack_id]+=1
            
            
    def remove(self, stack_id):
        if self.last_idx[stack_id] == self.first_idx[stack_id]:
            self.array[self.last_idx[stack_id]] = None
            return None
        else:
            data = self.array[self.last_idx[stack_id]]
            self.array[self.last_idx[stack_id]] = None
            self.last_idx[stack_id] -= 1
            return data
    
            
s = ThreeStackInOneArray(size = 9)
s.add(0,1)
s.add(1,2)
s.add(0,2)
s.add(0,4)
print(s.remove(0))
s.add(1,5)
s.add(2,104)
print(s.array)


# In[85]:


class LNode:
    def __init__(self, value):
        self.value = value
        self.next = None
    
    

def partition(list_head, value):
    head = list_head
    tail = list_head
    node = list_head.next
    while node is not None:
        new_node = LNode(node.value)
        if new_node.value < value:
            new_node.next = head
            head = new_node
        else:
            tail.next = new_node
            tail = new_node
        node = node.next
    return head

mylist = LNode(1)
head = mylist
mylist.next = LNode(8)
mylist = mylist.next
head.next = mylist
mylist.next = LNode(5)
mylist = mylist.next
mylist.next = LNode(3)
mylist = mylist.next
mylist.next = LNode(36)
mylist = mylist.next
mylist.next = LNode(2)
mylist = mylist.next


print(head.value, head.next.value, head.next.next.value, head.next.next.next.value, head.next.next.next.next.value, head.next.next.next.next.next.value)
new_head = partition(head, 5)
print(new_head.value, new_head.next.value, new_head.next.next.value, new_head.next.next.next.value, new_head.next.next.next.next.value, new_head.next.next.next.next.next.value)


# In[101]:



def is_palindrom(head):
    slow_itr = head
    fast_itr = head
    stack = Stack()
    stack.push(slow_itr.value)
    while fast_itr is not None and fast_itr.next is not None:
        fast_itr = fast_itr.next.next
        slow_itr = slow_itr.next
        stack.push(slow_itr.value)

    if fast_itr.next is None:
        stack.pop()
    slow_itr = slow_itr.next
    while slow_itr is not None:
        if slow_itr.value != stack.peek():
            return False
        stack.pop()
        slow_itr= slow_itr.next
    return True

mylist = LNode(1)
head = mylist
mylist.next = LNode(8)
mylist = mylist.next
head.next = mylist
mylist.next = LNode(2)
mylist = mylist.next
mylist.next = LNode(9)
mylist = mylist.next
mylist.next = LNode(5)
mylist = mylist.next
mylist.next = LNode(8)
mylist = mylist.next
mylist.next = LNode(1)
mylist = mylist.next


print(head.value, head.next.value, head.next.next.value, head.next.next.next.value, head.next.next.next.next.value, head.next.next.next.next.next.value, head.next.next.next.next.next.next.value)
print(is_palindrom(head))


# In[104]:


def urlify(st):
    length = 0
    for char in st:
        if char == ' ':
            length += 3
        else:
            length +=1
    array = [' ']*length
    
    count = 0
    for char in st:
        if char is ' ':
            array[count:count+3] = '%20'
            count+=3
        else:
            array[count] = char
            count+=1
    return array
print(urlify('Mr. John Smith '))


# In[106]:


def check_perm(st1, st2):
    counts = [0]*128
    for char in st1:
        counts[ord(char)] += 1
    for char in st2:
        counts[ord(char)] -= 1
    sum_all = sum(counts)
    if sum_all != 0:
        return False
    return True

check_perm('alexa', 'lexa')


# In[115]:


def is_palindrom_str(str1):
    for i in range(int(len(str1)/2)):
        print(str1[i] , str1[-1-i])
        if str1[i] != str1[-1-i]:
            return False
    return True

print(is_palindrom_str('abcdffdcba'))


# In[124]:


def str_compression(st):
    counts = [0] *128
    prev_char = None
    output = "{}".format(st[0])
    for char in st:
        idx = ord(char)
        counts[idx] += 1
        if prev_char is None:
            prev_char = char
        else:
            if prev_char != char:
                output = "{}{}{}".format(output, counts[ord(prev_char)],char)
                counts[ord(prev_char)] = 0
                prev_char = char
    return output

print(str_compression('aabcccccaaa '))


# In[134]:


def one_edit(str1, str2):
    str1_itr = 0
    str2_itr = 0
    len1 = len(str1)
    len2 = len(str2)
    if abs(len1 - len2) > 1:
        return False
    edit_count = 0 
    while str1_itr < len1 and str2_itr<len2:
        if str1[str1_itr] == str2[str2_itr]:
            str1_itr +=1
            str2_itr +=1
        else:
            edit_count+=1
            if edit_count > 1:
                return False
            if len1 < len2:
                str2_itr +=1
            elif len2 < len1:
                str1_itr +=1
            else:
                str1_itr +=1
                str2_itr +=1
    return True

print(one_edit('abc', 'acb'))


# In[154]:


def rotate_mat_90(mat):
    return [list(r) for r in zip(*mat[::-1])]
d = 3
mat1 = [[y*d+x+1 for x in range(d)] for y in range(d)]
print(mat1)
print(mat1[::-1])
mat2 = rotate_mat_90(mat1)
print(mat2)


# In[189]:


def set_zero(mat, m, n):
    zero_rows = [i for i , row in enumerate(mat) if not all(row)]
    print(zero_rows)
    zero_cols = [i for i, col in enumerate(zip(*mat)) if not all(col)]
    print(zero_cols)
    for row in zero_rows:
        mat[row] = [0]*m
    for col in zero_cols:
        for row_idx in range(n):
            mat[row_idx][col] = 0
    return mat
test_mat = [[1,2,0],[0,1,1],[1,1,1]]
print(all([1,4,1]))
print(set_zero(test_mat, 3, 3))


# In[188]:


for i in zip(*test_mat):
    print(i)
print('****')
result = [[None for i in range(3)] for j in range(3)]
print(result)
for i in range(3):
    for j in range(3):
        result[i][j] = test_mat[3-j-1][i]
print(result)


# In[ ]:



