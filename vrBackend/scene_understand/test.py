def max_essence_posts(n, m, k, intervals):
    posts = ["普通"] * n  # 初始帖子列表为全"普通"状态

    # 将区间内的帖子状态设置为"精华"
    for interval in intervals:
        li, ri = interval
        for i in range(li - 1, ri):
            posts[i] = "精华"

    # 遍历数组，计算每个长度为k的连续子数组中"精华"帖子的数量，并记录最大值
    max_essence = 0
    for i in range(n - k + 1):
        essence_count = 0
        for j in range(i, i + k):
            if posts[j] == "精华":
                essence_count += 1
        max_essence = max(max_essence, essence_count)

    return max_essence


# 输入示例
n, m, k = map(int, input().split())
intervals = []
for _ in range(m):
    interval = tuple(map(int, input().split()))
    intervals.append(interval)

# 输出结果
result = max_essence_posts(n, m, k, intervals)
print(result)
