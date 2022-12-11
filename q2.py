import random
import numpy as np
import scipy
from cvxpy import *
import matplotlib.pyplot as plt


def generate_offline_data(p_min, p_max, w_max, t):
	offline_data = []
	for i in range(t):
		v_t = random.randint(p_min, p_max)

		# print("current v, upper_b, lower_b:", v_t, v_t/p_max, v_t/p_min)
		# print("upper bound", np.minimum(v_t/p_min, w_max))
		w_t = random.uniform(np.maximum(0,v_t/p_max), np.minimum(v_t/p_min, w_max))
		# w_t = random.uniform(0, w_max)
		# print(w_t)
		# print(v_t/p_max, w_t)

		offline_data.append((v_t, w_t))
	return offline_data


def run_knapsack(p_min=1, p_max=100, w_max=0.9, t=100):
	# use offline data in online form

	y = 0
	print("generating data...")
	offline_data = generate_offline_data(p_min, p_max, w_max, t)
	print("solving online.....")
	beta_star = 1/ (1 + np.log(p_max / p_min))
	p_threshold = 0
	total_utility = 0

	for pair in offline_data:
		v_t, w_t = pair

		p_threshold = p_min if (y < beta_star) else p_min * np.exp(y/beta_star - 1)

		if (v_t / w_t) >= p_threshold:
			y += w_t 
			total_utility += v_t
	print("done...")
	x = solve_optimal_value(offline_data)
	cr = x / total_utility
	# print(x / total_utility)
	return cr

def solve_optimal_value(offline_data):
	v = []
	w = []
	for pair in offline_data:
		v_t, w_t = pair
		v.append(v_t)
		w.append(w_t)
	selection = cvxpy.Variable(len(w), boolean=True)
	weight_constraint = w * selection <= 1
	total_utility = v * selection
	knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_utility), [weight_constraint])

	return knapsack_problem.solve(solver=cvxpy.GLPK_MI)
	# The data for the Knapsack problem


if __name__ == '__main__':
	
	p_max_list = [10, 20, 40, 60, 100, 200, 400, 600, 1000]
	w_max_list = [0.00001, 0.00005, 0.001, 0.002, 0.005, 0.008 , 0.01, 0.05, 0.1]
	cr_s = []
		# current_cr = []
	for cur_w_max in w_max_list:
		total = 0
		for i in range(100):
			cr = run_knapsack(w_max=cur_w_max, t=1000)
			total += cr
		# print("average competitive ratio: ", (total / 10))
		cr_s.append((total / 100))

	# fig, ax = plt.subplots(figsize=(12,8))
	plt.plot(w_max_list, cr_s)
	# for index in range(len(p_max_list)):
 # 		ax.text(p_max_list[index], cr_s[index], cr_s[index], size=12)
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# # ax.set_ylim()
	# for i,j in zip(p_max_list,cr_s):
	# 	ax.annotate(str(j),xy=(i,j))

	plt.ylabel('infiniestimal CR')
	plt.savefig("q1_parameter_w_max.png")
	# plt.show()








