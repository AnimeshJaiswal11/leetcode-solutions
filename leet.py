import math
from collections import *
def first_non_repeat(s):
    char = []
    nums = []
    for i in range(len(s)):
        if s[i] not in char:
            char.append(s[i])
            nums.append(1)
        else:
            idx = char.index(s[i])
            nums[idx] += 1
    for i in range(len(nums)):
        if nums[i] == 1:
            return char[i]
    return 'No such non repeating character exists'

def isomorphic(s1,s2):
    d1 = dict()
    d2 = dict()
    for i in range(len(s1)):
        if s1[i] not in d1:
            d1[s1[i]] = s2[i]
        else:
            if d1[s1[i]] != s2[i]:
                return False
    for i in range(len(s2)):
        if s2[i] not in d2:
            d2[s2[i]] = s1[i]
        else:
            if d2[s2[i]] != s1[i]:
                return False
    return True

def remove_consecutive_duplicate(s):
    stack = []
    for i in range(len(s)):
        if len(stack) == 0:
            stack.append(s[i])
        else:
            if stack[-1] == s[i]:
                stack.pop()
            else:
                stack.append(s[i])
    return ''.join(stack)

def repeated_substring_pattern(s):
    for i in range(1,len(s)):
        if s == s[i:] + s[:i]:
            return True
    return False

def find_substring(s):
    d = dict()
    for i in range(len(s)):
        d[s[i]] = i
    res = []
    start = 0
    end = 0
    for i in range(len(s)):
        end = max(d[s[i]],end)
        if i == end:
            res.append(end - start + 1)
            start = i + 1
    return res


def merge_two_sorted_arrays(A, B):
    res = []
    i = 0
    j = 0
    while i < len(A) and j < len(B):
        if A[i] < B[j]:
            res.append(A[i])
            i += 1
        else:
            res.append(B[j])
            j += 1
    while i < len(A):
        res.append(A[i])
        i += 1
    while j < len(B):
        res.append(B[j])
        j += 1
    return res

def int_to_roman(n):
    num = [1,4,5,9,10,40,50,90,100,400,500,900,1000]
    syn = ['I','IV','V','IX','X','XL','L','XC','C','CD','D','CM','M']
    i = 12
    s = ''
    while i >= 0:
        q = n//num[i]
        n = n%num[i]
        while q > 0:
            s += syn[i]
            q -= 1
        i -= 1
    return s

def Atoi(s):
    intmin = -2**31
    intmax = 2**31-1
    res = 0
    if s[0] == '-':
        i = 1
        sign = -1
    else:
        i = 0
        sign = 1
    while i < len(s):
        if ord(s[i])-48 <= 9 and ord(s[i])-48 >= 0:
            res = res*10+(ord(s[i])-48)
            i += 1
        else:
            break
    res *= sign
    if res < intmin:
        return intmin
    elif res > intmax:
        return intmax
    else:
        return res

def max_prod_suarray(arr):
    if len(arr) == 0:
        return 'Empty array provided!!'
    cur_max = arr[0]
    cur_min = arr[0]
    ans = arr[0]
    prev_min = arr[0]
    prev_max = arr[0]
    for i in range(1,len(arr)):
        cur_max = max(prev_max*arr[i],prev_min*arr[i],arr[i])
        cur_min = min(prev_max*arr[i],prev_min*arr[i],arr[i])
        ans = max(ans,cur_max)
        prev_max = cur_max
        prev_min = cur_min
    return ans

def overlap_intervals(A):
    i = 0
    while i < len(A)-1:
        if A[i][1] >= A[i+1][0]:
            A[i][0] = min(A[i][0],A[i+1][0])
            A[i][1] = max(A[i][1],A[i+1][1])
            del A[i+1]
        else:
            i += 1
    return A

def insert_intervals(A,I):
    if len(A) == 0:
        return [I]
    if len(I) == 0:
        return A
    flag = False
    for i in range(len(A)):
        if I[1] < A[i][0]:
            A.insert(i,I)
            flag = True
            break
        if A[i][1] >= I[0]:
            A[i][0] = min(A[i][0],I[0])
            A[i][1] = max(A[i][1],I[1])
            flag = True
            break
    if flag == True:
        return overlap_intervals(A)
    else:
        A.append(I)
        return A

def wanna_rob(A):
    if len(A) < 3:
        return max(A)
    else:
        dp = [0]*len(A)
        dp[0] = A[0]
        dp[1] = max(A[0],A[1])
        for i in range(2,len(A)):
            dp[i] = max(dp[i-1],A[i]+dp[i-2])
    return dp[-1]

def sequemtial_digit(low,high):
    lenl = len(str(low))
    lenh = len(str(high))
    ans = []
    s = '123456789'
    while lenl <= lenh:
        i = 0
        while i + lenl <= 9:
            cur = int(s[i:i+lenl])
            if cur >= low and cur <= high:
                ans.append(cur)
            i += 1
        lenl += 1
    return ans

def sheen_temmo(A,d):    #### Time Complexity = O(n) and Sapce Complexity = O(1).......
    if len(A) == 0:
        return 0
    elif len(A) == 1:
        return d
    else:
        ans = 0
        for i in range(len(A)-1):
            if A[i+1] - A[i] < d:
                ans += A[i+1] - A[i]
            else:
                ans += d
    ans += d
    return ans

def sorted_squared_array(A):
    ans = []
    for i in range(len(A)):
        ans.append(A[i]*A[i])
    return sorted(ans)

def max_runs(k,n):
    ans = 0
    for i in range(1,n+1):
        if i % k == 0:
            ans += 1
        else:
            ans += 2
    return ans


def leader_element(A):
    ans = [A[-1]]
    i = len(A) - 2
    m = A[-1]
    while i >= 0:
        if A[i] > m:
            m = A[i]
            ans.insert(0,A[i])
        i -= 1
    return ans

def closest_number_to_divisible(n,m):
    rem = n % m
    if rem == 0:
        return n
    else:
        k = m - rem
        if k < rem:
            return n + k
        else:
            return n - rem

def str_evaluation(s):
    stack = [int(s[0])]
    i = 1
    while i < len(s):
        x = stack.pop()
        if s[i] == 'A':
            stack.append(x & int(s[i+1]))
        elif s[i] == 'B':
            stack.append(x | int(s[i + 1]))
        else:
            stack.append(x ^ int(s[i + 1]))
        i += 2
    return stack[0]

def min_efforts(A,B,X,Y):
    A.sort()
    B.sort()
    i = len(A) - 1
    eff = 0
    inhand = 0
    while i >= 0:
        if A == B and inhand == 0:
            break
        x = A[i] - B[i]
        if x > 0:
            eff += Y*x
            A[i] -= x
            inhand += x
        elif x < 0:
            eff += X*(-x)
            A[i] += x
            inhand += x
        i -= 1
    return eff

def number_of_recent_calls(t):
    queue = []
    ans = []
    for i in range(len(t)):
        if t[i] == None:
            ans.append(len(queue))
            continue
        if len(queue) > 0:
            while len(queue) > 0:
                if t[i] - queue[-1] > 3000:
                    queue.pop()
                else:
                    queue.insert(0,t[i])
                    break
        else:
            queue.append(t[i])
        ans.append(len(queue))
    return ans

def num_special(mat):
    n = len(mat)
    m = len(mat[0])
    row = [0] * n
    col = [0] * m
    for i in range(n):
        for j in range(m):
            if mat[i][j] == 1:
                row[i] += 1
                col[j] += 1
    count = 0
    for i in range(n):
        for j in range(m):
            if row[i] == 1 and col[j] == 1 and mat[i][j] == 1:
                count += 1
    return count

def max_ammo(W,w,n):
    cur  = w[0]
    dp = [0]*len(w)
    dp[0] = n[0]
    for i in range(1,len(w)):
        cur += w[i]
        if cur <= W:
            dp[i] = dp[i-1] + n[i]
        else:
            if w[i] <= W:
                if dp[i-1] <= n[i]:
                    dp[i] = n[i]
                else:
                    dp[i] = dp[i-1]
            else:
                dp[i] = dp[i-1]
    return dp[-1]

def makeGood(s: str) -> str:
    if len(s) == 0:
        return ''
    else:
        stack = []
        i = 0
        while i < len(s):
            if len(stack) == 0:
                stack.append(s[i])
            else:
                if s[i].isupper() and stack[-1].islower():
                    if s[i].lower() == stack[-1]:
                        stack.pop()
                    else:
                        stack.append(s[i])
                elif s[i].islower() and stack[-1].isupper():
                    if s[i].lower() == stack[-1]:
                        stack.pop()
                    else:
                        stack.append(s[i])
                else:
                        stack.append(s[i])
            i += 1
    return ''.join(stack)

#### subarray sum using prefix logic
def subarray_sum(A,k):
    pref = [0]*len(A)
    pref[0] = A[0]
    for i in range(1,len(A)):
        pref[i] = A[i] + pref[i-1]
    i = 0
    j = 1
    while j < len(A):
        if pref[j] - pref[i] == k:
            return A[i+1:j+1]
        elif pref[j] - pref[i] < k:
            j += 1
        else:
            i += 1
    return -1

def max_non_overlapping(A,k):
    pref = {}
    pref[0] = -1
    count = 0
    j = -1
    presum = 0
    for i in range(len(A)):
        presum += A[i]
        c = presum - k
        if c in pref and pref[c]>=j:
            count += 1
            j = i
        pref[presum] = i
    return count

def oranges(n):
    ans = set()
    q = [(n,0)]
    while len(q) > 0:
        x = q.pop()
        ans.add(x)
        if x[0] == 0:
            return x[1]
        else:
            if (x[0]-1,x[1]+1) not in ans:
                q.insert(0,(x[0]-1,x[1]+1))
            if x[0] % 3 == 0:
                ins = int(x[0]/3)
                if (ins,x[1]+1) not in ans:
                    q.insert(0,(ins,x[1]+1))
            if x[0] % 2 == 0:
                ins = int(x[0] / 2)
                if(ins, x[1] + 1) not in ans:
                    q.insert(0, (ins, x[1] + 1))

def aggressive_cow(A,c):
    low = 1
    n = len(A)
    high = n-1
    def works(A,c,guess,n):
        stall = 1
        prev_stall = 0
        for i in range(1,c):
            while A[stall] - A[prev_stall] < guess:
                stall += 1
                if stall == n:
                    return False
            prev_stall = stall
        return True
    ans = 0
    while low <= high:
        mid = int(low + (high-low)/2)
        if works(A,c,mid,n):
            low = mid + 1
            ans = max(ans,mid)
        else:
            high = mid - 1
    return ans

def most_visited_sector(n,round):
    if len(round) == 1:
        return round[0]
    else:
        freq = [0]*(n+1)
        for i in range(len(round) - 1):
            j = round[i]
            k = round[i+1]
            if j < k:
                while j < k:
                    freq[j] += 1
                    j += 1
            elif j > k:
                while j <= n:
                    freq[j] += 1
                    j += 1
                j = 1
                while j < k:
                    freq[j] += 1
                    j += 1
        freq[round[-1]] += 1
    m = max(freq)
    ans = []
    for i in range(len(freq)):
        if freq[i] == m:
            ans.append(i)
    return ans

def max_coins(A):
    A.sort()
    ans = 0
    i = 0
    j = len(A) - 1
    while i < j:
        mine = A[j-1]
        ans += mine
        j -= 2
        i += 1
    return ans

def find_latest_step(A,m):
    steps = 0
    s = '0'*(len(A))
    k = '1'*m
    for i in range(len(A)):
        s = s[:A[i] - 1] + '1' + s[A[i]:]
        steps += 1
        sam = s.split('0')
        print(sam)

def remove_covered_intervals(intervals):
    intervals = sorted(intervals, key = lambda x : (x[0],-x[1]))
    ans = 0
    end = -2**31
    for i in range(len(intervals)):
        if intervals[i][1] > end:
            end = intervals[i][1]
            ans += 1
    return ans

def complement(n):
    steps = 0
    num = n
    while n > 0:
        steps += 1
        n = n // 2
    ans = (1<<steps)-1
    return ans ^ num

def read_binary_watch(n):
    ans = []
    for i in range(12):
        for j in range(60):
            if bin(i).count('1') + bin(j).count('1') == n:
                time = str(i) + ':' + str(j).zfill(2)
                ans.append(time)
    return ans

def convert_hexadecimal(dec):
    if dec == 0:
        return '0'
    elif dec > 0:
        return hex(dec)[2:]
    else:
        return hex((1<<32)-(-dec))[2:]

def longest_palindrome(s):
    d = dict()
    for i in range(len(s)):
        if s[i] in d:
            d[s[i]] += 1
        else:
            d[s[i]]  = 1
    ans = 0
    flag = False
    for i in d:
        if d[i] % 2 ==0:
            ans += d[i]
        else:
            ans += 2*(d[i]//2)
            flag = True
    if flag:
        ans += 1
    return ans

def assign_cookies(g,s):
    i = 0
    j = 0
    count = 0
    while i < len(g) and j < len(s):
        if g[i] <= s[j]:
            count += 1
            i += 1
            j += 1
        else:
            j += 1
    return count

def isprime(n):
    i = 2
    while i*i <= n:
        if n % i == 0:
            return False
        i += 1
    return True

def prime_upto(n):
    s = [2,3]
    i = 4
    count = 2
    while count<n:
        if isprime(i):
            s.append(i)
            count += 1
        i += 1
    return s

def fibonacii(n):
    s= [0]*n
    s[0] = 1
    s[1] = 1
    for i in range(2,n):
        s[i] = s[i-1]+s[i-2]
    return s[-1]


print(prime_upto(10))
def series(n):
    sam = [1,2,1,3]
    if n < 4:
        return sam[n-1]
    if n % 2 == 0:
        return prime_upto(n//2)[-1]
    else:
        return fibonacii((n//2)+1)

def power(x):
    if x % 2 != 0:
        return 0
    else:
        m = 0
        i = 1
        while 2**i <= x//2:
            if x % (2**i) == 0:
                m = i
            i += 1
        return m

def max_power(s):
    new = s
    n = len(s)
    m = 0
    for i in range(len(s)):
        x = n - i
        cur = s[x:] + s[:x]
        cur = int(cur,2)
        m = max(power(cur),m)
    return m

def cut_sticks(A):
    n = len(A)
    ans = []
    d = Counter(A)
    for i in sorted(d.keys()):
        ans.append(n)
        n -= d[i]
    return ans

def non_divisible_subset(A,k):
    rem = [0]*k
    for i in range(len(A)):
        rem[A[i]%k] += 1
    ans = min(rem[0],1)
    if k % 2 == 0:
        ans += min(rem[k//2],1)
    for i in range(1,k//2+1):
        if i != k - i:
            ans += max(rem[i],rem[k-i])
    return ans

def rotation(A,k):
    if k == 0:
        return A
    k = k % len(A)
    i = 0
    n = len(A)
    j = n - k
    while j < n:
        A[i] , A[j] = A[j] , A[i]
        j += 1
        i += 1
    i = k
    j = n-k
    while i < n and j < n:
        A[i] , A[j] = A[j] , A[i]
        i += 1
        j += 1
    return A

def andaddxor(n):
    bits = int(math.log(n,2)) + 1
    zero = 0
    for i in range(bits):
        if n & 1<<i == 0:
            zero += 1
    return 2**zero

def pivot(A):
    n = len(A)
    if A[0] < A[n - 1]:
        return -1
    else:
        l = 0
        h = n - 1
        while l <= h:
            mid = int(l + (h - l) / 2)
            if mid > 0 and A[mid] > A[mid - 1] and mid < n and A[mid] > A[mid + 1]:
                return mid
            elif A[mid] >= A[l]:
                l = mid + 1
            elif A[mid] < A[l]:
                h = mid - 1
            else:
                return mid

def unique_binary_search_trees(n):
    dp = [0]*(n+1)
    dp[0] = 1
    dp[1] = 1
    dp[2] = 2
    for i in range(3,n+1):
        count = 0
        for j in range(i):
            count += dp[j] * dp[i-j-1]
        dp[i] = count
    return dp[-1]

def array_partioning(A,x):    #### Time Complexity = O(N) and Space Complexity = O(1) ...........
    i = 0
    j = 0
    while i < len(A) and j < len(A):
        if A[j] < x:
            temp = A[j]
            A[j] = A[i]
            A[i] = temp
            i += 1
            j += 1
        else:
            j += 1
    i = 0
    j = 0
    while i < len(A) and j < len(A):
        if A[i] < x:
            i += 1
            j += 1
        elif A[i] == x:
            if i < len(A)-1 and A[i+1] < x:
                A[i] , A[i+1] = A[i+1] , A[i]
                i += 1
                j += 1
            else:
                A[j] , A[i] = A[i] , A[j]
                i += 1
                j += 1
        else:
            i += 1
    return A

def subarraySum(nums, k: int) -> int:
    ref = set()
    pref = [0] * (len(nums) + 1)
    for i in range(len(nums)):
        pref[i + 1] = pref[i] + nums[i]
    ans = 0
    for i in range(len(pref)):
        if (pref[i]-k) in ref:
            ans += 1
        ref.add(pref[i])
    return ans

def countSquares(matrix) -> int:
    m = len(matrix)
    n = len(matrix[0])
    dp = [[0] * (n + 1)] * (m + 1)
    ans = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == 0:
                dp[i][j] = 0
            else:
                c = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
                dp[i][j] = 1 + c
                ans += 1 + c
    return ans

def counting_elements(arr):
    ans = 0
    d = Counter(arr)
    for i in d:
        if i+1 in d:
            ans += d[i]
    return ans

def all_paths_mat(start,end,A,s,n,m):
    if start == end:
        A.append(s)
        return
    if start[0] >= n or start[1] >= m:
        return
    all_paths_mat([start[0]+1,start[1]],end,A,s+"V",n,m)
    all_paths_mat([start[0],start[1]+1],end,A,s+"H",n,m)
    return A

def all_paths_mat_restrictions(mat,start,end,A,s,n,m):
    if start == end:
        A.append(s)
        return
    if start[0] >= n or start[1] >= m or mat[start[0]][start[1]] == 0:
        return
    all_paths_mat_restrictions(mat,[start[0]+1,start[1]],end,A,s+"V",n,m)
    all_paths_mat_restrictions(mat,[start[0],start[1]+1],end,A,s+"H",n,m)
    return A

def number_of_island(mat):
    if not mat:
        return 0
    def dfs(mat,i,j):
        if i < 0 or j < 0 or i >= len(mat) or j >= len(mat[0]) or mat[i][j] != 1:
            return
        mat[i][j] = 0
        dfs(mat,i+1,j)
        dfs(mat,i-1,j)
        dfs(mat,i,j+1)
        dfs(mat,i,j-1)
    ans = 0
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 1:
                ans += 1
                dfs(mat,i,j)
    return ans

def max_sub_adj(arr):
    def recursive(arr,i):
        if i == len(arr)-1:
            return arr[i]
        if i >= len(arr):
            return 0
        a = recursive(arr,i+1)
        b = recursive(arr,i+2) + arr[i]
        return max(a,b)

    def dynamic(arr):
        if len(arr) == 0:
            return None
        if len(arr) <= 2:
            return max(arr)
        A = [0]*(len(arr))
        A[0] = arr[0]
        A[1] = arr[1]
        for i in range(2,len(arr)):
            A[i] = max(arr[i]+A[i-2],A[i-1])
        return A[-1]
    return dynamic(arr)

def tower_height(heights,k):
    if len(heights) < 2:
        return 0
    mini = min(heights[0],heights[1])
    maxi = max(heights[0],heights[1])
    ans = abs(maxi-mini-2*k)
    for i in range(2,len(heights)):
        if heights[i] > maxi:
            maxi = heights[i]
            ans = max(abs(maxi-mini-2*k),ans)
        if heights[i] < mini:
            mini = heights[i]
            ans = max(abs(maxi-mini-2*k),ans)
    return ans

def fibonacci_memoization(n):     #### top down approach with memoization ........
    fib = [0]*(n+1)
    if n < 2:
        return n
    fib[0] = 0
    fib[1] = 1
    def fibon(fib,n):
        if fib[n] != 0 or n == 0:
            return fib[n]
        fib[n] = fibon(fib,n-1)+fibon(fib,n-2)
        return fib[n]
    return fibon(fib,n),fib

def fibonacci_tabulation(n):     #### bootom up approach with tabulation ........
    fib = [0]*(n+1)
    if n < 2:
        return n
    fib[0] = 0
    fib[1] = 1
    for i in range(2,n+1):
        fib[i] = fib[i-1] + fib[i-2]
    return fib[n],fib

def frequencySort(nums):
    A = [0] * 202
    for i in range(len(nums)):
        A[nums[i] + 100] += 1
    C = []
    for i in range(len(A)):
        C.append([i - 100, A[i]])
    C.sort(key = lambda x: (x[1],-x[0]))
    ans = []
    for c in C:
        f = [c[0]]*c[1]
        ans += f
    return ans

def knapsack(weight,prices,bag,n):
    def knapsack_dp(wt,val,W,n):
            K = [[0 for x in range(W + 1)] for x in range(n + 1)]

            # Build table K[][] in bottom up manner
            for i in range(n + 1):
                for w in range(W + 1):
                    if i == 0 or w == 0:
                        K[i][w] = 0
                    elif wt[i - 1] <= w:
                        K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
                    else:
                        K[i][w] = K[i - 1][w]
            return K[n][W]
    return knapsack_dp(weight,prices,bag,n)

def trap(height) -> int:
    if len(height) < 3:
        return 0
    left = [height[0]]
    right = [0] * len(height)
    right[-1] = height[-1]
    for i in range(1, len(height)):
        if left[i - 1] > height[i]:
            left.append(left[i - 1])
        else:
            left.append(height[i])
    for i in range(len(height) - 2, -1, -1):
        if right[i + 1] > height[i]:
            right[i] = right[i + 1]
        else:
            right[i] = height[i]
    ans = 0
    for i in range(1, len(height) - 1):
        ans += min(left[i], right[i]) - height[i]
    return ans

def subset_partition(arr):
    s = sum(arr)
    if s % 2 != 0:
        return False
    def recursive(arr,s,n):
        if n == 0 and s != 0:
            return False
        if s == 0:
            return True
        if arr[n-1] > s:
            return recursive(arr,s,n-1)
        else:
            return recursive(arr,s,n-1) or recursive(arr,s-arr[n-1],n-1)
    return recursive(arr,s//2,len(arr))

def reverse_bits(n):
    ans = 0
    for i in range(32):
        cur = n ^ (1<<i)
        cur = cur & (1<<i)
        ans = ans | cur
    return ans

def permuatations(nums):
    res = []
    def permu_helper(nums, res, path):
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            permu_helper(nums[:i] + nums[i + 1:], res, path + [nums[i]])
    permu_helper(nums,res,[])
    return res

def perfect_square(n):
    sqr = int(math.sqrt(n))
    squares = []
    for i in range(1,sqr+1):
        squares.append(i*i)
    start = len(squares)-1
    dp = [[0]*(n+1) for i in range(start+2)]
    def helper(count,n,squares,start):
        if n == 0:
            return count
        if start < 0:
            return 150
        if squares[start] > n:
            return helper(count,n,squares,start-1)
        else:
            return min(helper(count+1,n-squares[start],squares,start),helper(count+1,n-squares[start],squares,start-1),helper(count,n,squares,start-1))
    return helper(0,n,squares,start)

def perfect_square_dp(n):
    dp = [0]*(n+1)
    dp[1] = 1
    sq = []
    for i in range(1,int(math.sqrt(n))+1):
        sq.append(i*i)
    fix = 1
    for i in range(2,n+1):
        dp[i] += 1
        cur = i
        if cur in sq:
            fix = cur
        cur = cur-fix
        dp[i] += dp[cur]
    return dp[n]

print(perfect_square_dp(127))


def closeStrings(word1: str, word2: str) -> bool:
    d = Counter(word1)
    f = Counter(word2)
    a = []
    b = []
    c,e = "",""
    for i in d:
        a.append(d[i])
        c += i
    for i in f:
        b.append(f[i])
        e += i
    a = sorted(a)
    b = sorted(b)
    c = sorted(c)
    e = sorted(e)
    return a == b and c == e

def letterCombinations(digits):
    d = {2: "abc", 3: "def", 4: "ghi", 5: "jkl", 6: "mno", 7: "pqrs", 8: "tuv", 9: "wxyz"}
    def helper(ans,start,n,dig,d,cur):
        if start == n:
            ans.append(cur)
            return
        for i in range(len(d[int(dig[start])])):
            cur += d[int(dig[start])][i]
            helper(ans,start+1,n,dig,d,cur)
            cur = cur[:-1]
        return ans
    n = len(digits)
    return helper([],0,n,digits,d,"")

def transpose_matrix(mat):
    k = len(mat)
    m = 0
    n = 0
    while m < k and n < k:
        i = m
        j = m
        while i < k and j < k:
            mat[i][m] , mat[n][j] = mat[n][j] , mat[i][m]
            i += 1
            j += 1
        m += 1
        n += 1
    return mat

def next_greater(A):
    ans = [0]*len(A)
    stack = []
    for i in range(len(A)-1,-1,-1):
        while stack and stack[-1] <= A[i]:
            stack.pop()
        if len(stack) > 0:
            ans[i] = stack[-1]
        else:
            ans[i] = -1
        stack.append(A[i])
    return ans

def nearest_smaller(A):
    ans = [0]*len(A)
    stack = []
    for i in range(len(A)):
        while stack and stack[-1] >= A[i]:
            stack.pop()
        if not stack:
            ans[i] = -1
        else:
            ans[i] = stack[-1]
        stack.append(A[i])
    return ans

def first_negative(A,k):
    dum = []
    for i in range(len(A)):
        if A[i] < 0:
            dum.append((A[i],i))
    ans = []
    for i in range(len(A)-k+1):
        j = i+k
        flag = False
        for num,idx in dum:
            if idx >= i and idx < j:
                flag = True
                ans.append(num)
                break
            if idx >= j:
                break
        if not flag:
            ans.append(0)
    return ans

def rod_cutting(prices):
    l = [i for i in range(1,len(prices)+1)]
    dp = [[0]*(len(prices)+1) for _ in range(len(prices))]
    for i in range(1,len(prices)):
        for j in range(1,len(prices)):
            if l[i-1] <= j:
                dp[i][j] = max(prices[i-1] + dp[i][j-l[i-1]], dp[i][j])
            else:
                dp[i][j] = dp[i][j]
    return dp[-1][-1]

def gas_station(f,c):
    n = len(c)
    for i in range(n):
        start = i
        if f[i] < c[i]:
            continue
        else:
            cur = 0
            j = i
            while j < n:
                cur += f[i] - c[i]
                j += 1
            j = 0
            while j < i:
                cur += f[i]-c[i]
                j += 1
            if cur >= 0:
                return start
    return -1

