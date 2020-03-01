f = open("diff/per.txt", "r+")
nums = f.readlines()
nums = [int(i) for i in nums]
print(max(nums))
print(nums)
f.seek(0)
f.truncate()
