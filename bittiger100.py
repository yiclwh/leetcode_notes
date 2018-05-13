'''
33. Search in Rotated Sorted Array
'''
def search(nums, target):
    """
    :type nums: List[int]  把4种情况都写出来，对照着2X2写出来就可以了
    :type target: int 
    :rtype: int 
    """
    start, end = 0, len(nums) - 1
    while start <= end:
        mid = start + (end - start) //2
        if nums[mid] == target:
            return mid
        elif target > nums[mid]:
            if nums[mid] < nums[start] and target > nums[end]:
                end = mid - 1
            else:
                start = mid + 1
        elif target < nums[mid]:
            if nums[mid] > nums[end] and target < nums[start]:
                start = mid + 1
            else:
                end = mid - 1
    return -1

'''
38. Count and Say
1.     1
2.     11
3.     21
4.     1211
5.     111221
'''
def countAndSay(self, n):
    """
    :type n: int 照常理推出来,从第一位往后看,最后一位只需要res += str(count) + s[-1]
    :rtype: str  因为无论怎么样都已经count过了
    """
    if n == 0:
        return ''
    elif n == 1:
        return '1'
    else:
        s = self.countAndSay(n -1)
        res, count = '', 1
        for i in range(len(s) - 1):
            if s[i] == s[i+1]:
                count += 1
            else:
                res += str(count) + s[i]
                count = 1
        res += str(count) + s[-1]
    return res

'''
39. Combination Sum
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
'''
def combinationSum(candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    res = []
    helper(candidates, target, 0, [], res)
    return res
    
def helper (nums, target, start, comb, res):
    if target == 0:
        res.append(comb)
    elif target > 0:
        for i in range(start, len(nums)):
            helper(nums, target - nums[i], i, comb + [nums[i]], res)

'''
41. First Missing Positive
'''
def firstMissingPositive(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    length = len(nums)
    for i in range(length):
        while 1 <= nums[i] <= length and nums[nums[i] - 1] != nums[i]:
            temp = nums[i]
            nums[i] = nums[temp - 1]
            nums[temp - 1] = temp
            
    for j in range(length):
        if nums[j] != j + 1:
            return j + 1
    return length + 1

'''
49. Group Anagrams 这里用defaultdict 也可以不用 就多一个if语句
'''
def groupAnagrams(self, strs):
    """
    :type strs: List[str]
    :rtype: List[List[str]]
    """
    from collections import defaultdict
    record = defaultdict(list)
    for s in strs:
        record[str(sorted(s))].append(s)
    return record.values()

'''
53. Maximum Subarray    其实有隐藏要求是必须要取一个数(不能得到0)
Input: [-2,1,-3,4,-1,2,1,-5,4],  最优解 应该先想到的是DP
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
'''
def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return 0
    prev = res = nums[0]
    for num in nums[1:]:
        prev = max(prev + num, num)
        res = max(prev, res)
    return res

'''
54. Spiral Matrix  轮圈打印 pop reverse 搞定
'''
def spiralOrder(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    res = []
    while matrix:
        res += matrix.pop(0)
        res += [x.pop() for x in matrix if x]   
        if matrix:
            res += reversed(matrix.pop())
        res += [x.pop(0) for x in reversed(matrix) if x]
    return res

'''
55. Jump Game 
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
'''
def canJump(self, nums):
    """
    :type nums: List[int]   记录下当前能跳到的最大值,再去判断是不是终点
    :rtype: bool
    """
    reach= 0     
    for i in range(len(nums)):
        if i <= reach:
            reach = max(reach, i + nums[i])
            if reach >= len(nums) - 1:
                return True
    return False

'''
56. Merge Intervals 先sort 
'''
 def merge(self, intervals):
    """
    :type intervals: List[Interval]    再看当前的start在不在前一个的end前面
    :rtype: List[Interval]             注意可以只用一个array搞定.如果为空就append,查就查[-1]位
    """
    res = []
    intervals.sort(key = lambda x:x.start)
    for i in intervals:
        if res and i.start <= res[-1].end:
            res[-1].end = max(i.end, res[-1].end)
        else:
            res.append(i)
    return res

'''
62. Unique Paths
'''
def uniquePaths(self, m, n):
    """
    :type m: int   注意的点是 row 和 col 
    :type n: int   一般的定义都是 m x n 也就是 col x row
    :rtype: int     record[行][列] 也就是 record[b][a] record[n][m]
    """
    cols = [1 for x in range(m)]
    
    for r in range(n)[1:]:
        for i in range(m)[1:]:
            cols[i] += cols[i-1]
    return cols[m-1]

'''
72. Edit Distance
'''
def minDistance(self, word1, word2):
    """
    :type word1: str  要注意index = length - 1
    :type word2: str
    :rtype: int
    """
    l1, l2 = len(word1), len(word2)
    if not l1:
        return l2
    if not l2:
        return l1
    record = [[0 for i in range(l1 + 1)] for j in range(l2 + 1)]
    for i in range(l1 + 1):
        record[0][i] = i
    for i in range(l2 + 1):
        record[i][0] = i
    for i in range(l2 + 1)[1:]:
        for j in range(l1 + 1)[1:]:
            if word2[i-1] == word1[j-1]:
                record[i][j] = record[i-1][j-1]
            else:
                record[i][j] = min(record[i-1][j-1], record[i][j-1], record[i-1][j]) + 1
    return record[l2][l1]