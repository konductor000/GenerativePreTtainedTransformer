def solve(n, a):
    


def read(inputs):
    n = int(input())
    a = [int(i) for i in input().split()]
    inputs.append((n, a))


inputs = []
t = int(input())
for _ in range(t):
    read(inputs)

for i in range(t):
    solve(inputs[i])