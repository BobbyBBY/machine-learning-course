mode=13
mode_bin = bin(mode).replace('0b','')
print(mode_bin)
# 反转字符串
mode_bin = mode_bin[::-1]
mode_bin = list(mode_bin)
print(mode_bin)
mode_bin2 = mode_bin.copy()
mode_bin2[0] = 3
print(mode_bin)
print(mode_bin2)