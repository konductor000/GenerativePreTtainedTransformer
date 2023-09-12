def solve(n):
    left = 1
    right = n + 2
    ans = 10 ** 20

    while left <= right:
        mid = (left + right) // 2
        if mid * (mid-1) // 2 > n:
            right = mid - 1
        else:
            left = mid + 1
            ans = min(ans, mid+n-mid*(mid-1)//2)
    
    print(ans)


def read(inputs):
    n = int(input())
    inputs.append(n)


inputs = []
t = int(input())
for _ in range(t):
    read(inputs)

for i in range(t):
    solve(inputs[i])