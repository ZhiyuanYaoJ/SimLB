import numpy as np

class Cons_hash:
	def __init__(self, _M=256, _N=10, _perm = [], _C = 2):
		self.M = _M
		self.N = _N
		self.C = _C
		self.perm = _perm
		self.lookup = []
		self.nextIndex = []
		self.dip_list = [i for i in range(self.N)]

	def compute_perm(self):
		p = []
		for i in range(self.N):
			p.append(np.random.permutation(self.M).tolist())
		self.perm = p

	def update_N(self, N):
		self.N = N

	def update_M(self, M):
		self.M = M

	def update_dip_table(self,remove=[],add=[]):
		cnt = 0
		if remove != []:
			for i in remove:
				if i in self.dip_list:
					self.dip_list.remove(i)
					cnt -=1
		if add != []:
			add.sort()
			for i in add:
				if i not in self.dip_list:
					self.dip_list.insert(int(i),i)
					cnt +=1
		#self.N += cnt
		#if self.N <0:
		#	print("ERROR\n DELETED ALL THE ROWS IN THE PERMUTATION")

	def compute_table(self):
		if len(self.perm) == 0:
			print("ERROR\nPERMUTATION NOT INITIALIZED")
			return
		n = 0
		self.lookup = [[-1,-1] for i in range(self.M)]
		self.nextIndex = [0 for i in range(self.N)]

		while True:
			for i in self.dip_list:
				cut = False
				if self.nextIndex[i] >= self.M:
					continue
				c = self.perm[i][self.nextIndex[i]]
				while self.lookup[c][self.C-1]>= 0:
					self.nextIndex[i] += 1
					if self.nextIndex[i] == self.M:
						cut = True
						break
					c = self.perm[i][self.nextIndex[i]]
				if cut == True:
					continue
				choice = 0
				while self.lookup[c][choice] >=0:
					choice += 1
				self.lookup[c][choice] = i
				self.nextIndex[i] += 1
				n +=1
				if n == self.M*self.C:
					return self.lookup
