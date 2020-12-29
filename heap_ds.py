## Heaps are special type of tree data structure which is a complete binary tree    ........
def build_max_heap(A):
    n = len(A)//2 - 1
    def helper(A,n,i):
        l = 2*i+1
        r = 2*i+2
        if l < len(A) and A[l] > A[i]:
            large = l
        else:
            large = i
        if r < len(A) and A[r] > A[i]:
            large = r
        if large != i:
            A[large],A[i] = A[i],A[large]
            helper(A,n,large)
    for i in range(n,-1,-1):
        helper(A,n,i)
    for i in range(n-1,0,-1):
        A[0],A[i] = A[i],A[0]
        helper(A,i,0)
    return A

def build_min_heap(A):
    n = len(A)//2 - 1
    def helper(A,i):
        l = 2*i+1
        r = 2*i+2
        if r < len(A) and A[r] < A[i]:
            small = r
        else:
            small = i
        if l < len(A) and A[l] < A[i]:
            small = l
        if small != i:
            A[small],A[i] = A[i],A[small]
            helper(A,small)
    for i in range(n,-1,-1):
        helper(A,i)
    return A



