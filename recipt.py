total = int(input())
type_num = int(input())
calcurate = []
for i in range(type_num):
    a, b = map(int,input().split())
    calcurate.append(a*b)


if sum(calcurate) == total:
    print('Yes')
else: 
    print('No')
