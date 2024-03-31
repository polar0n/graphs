from singletons import Inf


class Node:
    def __init__(self, value, prev=None):
        self.value = value
        self.prev = prev


class Queue:
    '''Implement a first-in first-out(FIFO) queue.'''
    def __init__(self, initial_value=None):
        self.first = None       # front
        self.last = None        # rear
        if initial_value:
            self.put(initial_value)


    def put(self, value) -> None:       # Enqueue
        enqueue_node = Node(value)
        if self.empty():
            self.first = enqueue_node   # Declare the new node as first and last
            self.last = self.first
            return
        self.last.prev = enqueue_node   # Declare the previous node of the last node as the new node
        self.last = enqueue_node        # then, declare the new last node as the previous node of the last node


    def get(self):                      # Dequeue
        if self.empty():
            return
        value = self.first.value        # Save the value of the first
        self.first = self.first.prev    # Set the new first node
        return value


    def empty(self) -> bool:
        return not self.first


class Stack:
    '''Implement a last-in first-out(LIFO) queue.'''
    def __init__(self, initial_value=None):
        self.top = None
        if initial_value:
            self.push(initial_value)


    def push(self, value) -> None:
        new_node = Node(value)
        if self.empty():
            self.top = new_node
            return
        new_node.prev = self.top
        self.top = new_node


    def pop(self):
        if self.empty():
            return
        value = self.top.value
        self.top = self.top.prev
        return value


    def empty(self) -> bool:
        return not self.top


    def get_list(self) -> list:
        if self.empty():
            return list()
        
        l = list()
        current_node = self.top
        while current_node:
            l.append(current_node.value)
            current_node = current_node.prev
        
        return l
        

    def __str__(self) -> str:
        return f'Stack({str(self.get_list())[1:-1]})'


    def __repr__(self) -> str:
        return self.__str__()


class MinHeap:
    def __init__(self, key:str, items:list=None) -> None:
        '''
        MinHeap constructor. `key` parameter stands for the attribute name that stores the number
        which is used in making comparisons and structuring the objects in the heap. Each inserted object
        MUST have the `key` attribute; otherwise, `.insert` method will raise `AttributeError` and the heap
        cannot be structured. The `items` parameter represents the initial list of items that the heap will
        populate; providing an `items` parameter will invoke the `.heapify` method.
        '''
        self.heap = list()
        self.key = key
        self._last_elem_index = -1
        self._get_key = lambda object: getattr(object, key)
        if items:
            self.heapify(items)


    def insert(self, object):
        if not hasattr(object, self.key):
            raise AttributeError(f'Cannot complete insertion because object ({object}) \
                                   does not have attribute .{self.key}.')
        self.heap.append(object)
        self._last_elem_index += 1
        # Check for violation
        violation_index = self._last_elem_index
        while True:
            parent_index = self._get_parent_index(violation_index)
            # print(f"{parent_index = }")
            # print(f'{violation_index = }')
            if self._get_elem_key(parent_index) <= self._get_key(object):
                break
            # Swap the new object with its parent
            self._swap_elements(violation_index, parent_index)
            # Assume that the swap created a violation
            violation_index = parent_index
            # and let the next iteration take care of verifying the swap and swapping again


    def _extract_and_repair(self, index:int) -> object:
        '''
        Private method that extracts an object from the heap, replaces it with the last added
        element to the heap. Then it removes all the violations in the heap by replacing the
        violating root-node with the smallest of its child-nodes.
        '''
        if not (0 <= index < len(self.heap)):
            return
        extraction = self.heap[index]
        self.heap[index] = self.heap[self._last_elem_index]
        del self.heap[self._last_elem_index]
        self._last_elem_index -= 1

        current_root_index = index
        current_root_key = self._get_elem_key(current_root_index)
        while True:
            children_indices = self._get_children_indices(current_root_index)
            # Check if the current_root_node has any children
            if children_indices[0] > self._last_elem_index:
                break

            child0_key = self._get_elem_key(children_indices[0])
            child1_key = self._get_elem_key(children_indices[1])
            # Check for violations
            if current_root_key < child0_key and current_root_key < child1_key:
                break

            # Check if the left child is bigger than the right to determine which one is bigger
            left_is_bigger = int(child0_key > child1_key)
            # If it is not bigger (False -> 0) swap the current root node with the smallest of its children
            smallest_child_index = children_indices[left_is_bigger]
            self._swap_elements(smallest_child_index, current_root_index)
            # Update the current_root_index to be the new swapped index
            current_root_index = smallest_child_index
            current_root_key = self._get_elem_key(current_root_index)
            # and let the next iteration check for violations and swap if necessary

        return extraction


    def extract_min(self) -> object:
        '''
        Extract the node with the smallest value.
        '''
        return self._extract_and_repair(0)


    def delete(self, object:object) -> None:
        '''Remove the specified `object` from the heap.'''
        self._extract_and_repair(self.heap.index(object))


    def _get_elem_key(self, index:int) -> object:
        '''
        Returns the key of any object with index `index` in the heap.
        '''
        return self._get_key(self.heap[index]) if index <= self._last_elem_index else Inf()


    def size(self) -> int:
        return self._last_elem_index + 1


    def _get_parent_index(self, child_index: int) -> int:
        return ((child_index + 1) >> 1) - 1 if child_index > 0 else 0


    def _get_children_indices(self, parent_index: int) -> tuple[int]:
        '''
        Returns the indices of the children of a parent with `parent_index`.
        '''
        child_index = ((parent_index + 1) << 1) - 1
        return (child_index, child_index + 1)


    def _swap_elements(self, index1:int, index2:int):
        old_elem_index1 = self.heap[index1]
        self.heap[index1] = self.heap[index2]
        self.heap[index2] = old_elem_index1

    
    def empty(self) -> bool:
        return not self.size()
    

    def heapify(self, items: list):
        '''
        Transform the list of `items` into a heap object and update the current heap.
        '''
        for item in items:
            self.insert(item)        


if __name__ == '__main__':
    from random import randint
    h = MinHeap('value')
    array = (randint(1, 1000) for i in range(100))
    # array = range(30)[::-1]
    for value in array:
        h.insert(Node(value))
    heap_array = list(map(lambda node: node.value, h.heap))
    print(f'Heap: {heap_array}')

    # print(f'Minimal extraction: {h.extract_min().value}')
    # print(f'Heap: {list(map(lambda node: node.value, h.heap))}')
    old_value = h.extract_min().value
    h._extract_and_repair(5)
    new_value = None
    while not h.empty():
        new_value = h.extract_min().value
        # print(f'Minimal extraction: {h.extract_min().value}')
        # print(f'Heap: {list(map(lambda node: node.value, h.heap))}')
        if old_value <= new_value:
            old_value = new_value
            continue
        print(f'{old_value = } > {new_value = }')
