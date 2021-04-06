# 一个int数二进制中1的个数
def number_of_1(n):
    count = 0
    while(n):
        n = n & (n-1)
        count += 1
    return count

# int数转二进制数组
def int_to_str( n):
    n_bin = bin(n).replace("0b", "")
    # 反转字符串
    n_bin = n_bin[::-1]
    return n_bin

# 一维映射函数
def mapping(x, y, n):
    pass