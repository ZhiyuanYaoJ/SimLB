import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


N_ASS = 20

name = "../data/simulation/first-impression/1lb-20as-1worker-1stage-exp-0.50cpumu/hermes/rate0.845/test.log"

file = open(name,"r")

dic = {}

inp = []
for i in range(N_ASS):
	dic[i] = 0

cnt_c = 1
cnt_p = 1

_print = False

lookup = True

for line in file:
	if lookup and "////" in line:
		print(line)
		lookup = False

	if "----" in line:
		s = line.split("[")[1]
		c =list([int(i) for i in list(s.replace(']','').replace('\n','').split(","))])
		print(cnt_c, end=" ")
		for l in range(len(c)):
			print(" {} - {} |".format(l,c[l]),end="")
		print()
		inp.append(c)
		cnt_c +=1

	if "<<<<" in line:
		s = line.split("[")[1]
		c =list([int(i) for i in list(s.replace(']','').replace('\n','').split(","))])
		print("   SCORE ==>", end=" ")
		for l in range(len(c)):
			print(" {} - {} |".format(l,c[l]),end="")
		print()

	if ">>>>" in line:
		s = line.split("[")[1]
		c =list([int(i) for i in list(s.replace(']','').replace('\n','').split(","))])
		s = c
		for i in range(len(s)):
			dic[i] = int(s[i])

		plt.bar(list(dic),list(dic.values()))
		plt.savefig('plots/tmp' + str(cnt_p)  +'.png')
		cnt_p += 1
		_inp = False




	if "======" in line:
		_print = True


file.close()


#for k in dic:
#	print("{} - {}".format(k,dic[k]))

