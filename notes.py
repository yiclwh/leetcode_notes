'''
1 Matrix DP
    1. Unique paths I/II
    2. 过河
    3. Triangle 
    4. Minimum Path Sum

2 Sequence DP
    1. Climbing Stairs
     2. Jump game/Jump game II
    3. Palindrome Partitioning II
     4. Word Break
     5. Decode ways
    6. Longest Increasing Subsequence

3. Two Sequence DP
    1. Distince Subsequence 
    2. Edit Distance
    3. Interleaving String
    4. Longest Common Subsequence

Clues 如何想到使用DP

1.Find a max/min result
2.Decide whether something is posible
3.Count all possible solutions

动态规划的要素
1. status   //matrix      : f[i][j] 从1,1走到i,j ...
            //Sequence    : f[i] 前i个
            //2 sequence  : f[i][j] 前i匹配上前j个
            //Interval    : f[i][j] 表示区间i-j

2. transfer  // LCS； f[i][j] = max[f[i-1][j], f[i][j-1], f[i-1][j-1] + 1]
             // LIS: f[i] = max(f[j] + 1, a[i] >= a[j])
             // 分析最后一次划分 / 最后一个字符/最后***

3. initialize   // f[i][0]  f[0][i]
                // f[0]
                // LIS: f[1...n] = 1;

4. answer   // LIS: max{f[i]}
            // LCS: f[n][m]

5. loop     // Interval: 区间从小到大，先枚举区间长度。Palindrome Patitioning II




'''
# Jump game
def canJump(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    reach = 0
    for index, step in enumerate(nums):
        if reach >= index:
            reach = max(reach, index + step)
    return reach >= len(nums) - 1

# Longest Increasing Subsequence
def lengthOfLIS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return 0
    record = [1 for i in range(len(nums))]
    for i in range(len(nums))[1:]:
        for j in range(i):
            if nums[i] > nums[j]:
                record[i] = max(record[i], record[j] + 1)
    print(record)
    return max(record)

# nlogn solution
'''
tails is an array storing the smallest tail of all increasing subsequences with length i+1 in tails[i].
For example, say we have nums = [4,5,6,3], then all the available increasing subsequences are:

len = 1   :      [4], [5], [6], [3]   => tails[0] = 3
len = 2   :      [4, 5], [5, 6]       => tails[1] = 5
len = 3   :      [4, 5, 6]            => tails[2] = 6
We can easily prove that tails is a increasing array. Therefore it is possible to do a binary search in tails array to find the one needs update.

Each time we only do one of the two:

(1) if x is larger than all tails, append it, increase the size by 1
(2) if tails[i-1] < x <= tails[i], update tails[i]
Doing so will maintain the tails invariant. The the final answer is just the size.
'''

def lengthOfLIS(self, nums):
    tails = [0] * len(nums)
    size = 0
    for x in nums:
        i, j = 0, size
        while i != j:
            m = (i + j) / 2
            if tails[m] < x:
                i = m + 1
            else:
                j = m
        tails[i] = x
        size = max(i + 1, size)
    return size


# Two Sum binary Tree
# recursive solution
def twoSum(self, root, n):
    # write your code here
    def helper(node, k, record):
        if node:
            if node.val in record:
                return [node.val, n - node.val]
            else:
                record.add(n - node.val)
            return helper(root.left, n, record) or helper(root.right, n, record)
    
    return helper(root, n, set())


# DFS interative
def twoSum(self, root, n):
    # write your code here
    stack = [root]
    record = set()
    while stack:
        node = stack.pop()
        if node:
            if node.val in record:
                return [node.val, n - node.val]
            else:
                record.add(n - node.val)
            stack.extend([node.left, node.right])