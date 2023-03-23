from copy import deepcopy as copy
import numpy as np
import os
import time
from functools import partial
from disp import get_ordered_colors
from aux import gaussian_if_under_val, exp_if_under_val, rev_argsort, set_smallest_n_zero
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import multiprocessing as mp
import argparse
import cma
from numba import njit
from scipy.sparse import csc_matrix
from csv_reader import read_csv
from csv_writer import write_csv

from rate_network import simulate, tanh, generate_gaussian_pulse

### Parse arguments 

parser = argparse.ArgumentParser()
parser.add_argument('--std_expl', metavar='std', type=float, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--l1_pen', metavar='l1', type=float, help='Prefactor for L1 penalties on loss function')
parser.add_argument('--pool_size', metavar='ps', type=int, help='Number of processes to start for each loss function evaluation')
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')
parser.add_argument('--fixed_data', metavar='fd', type=int, help='')
parser.add_argument('--load_initial', metavar='li', type=str, help='File from which to load the best params as an initial guess')
parser.add_argument('--seed', metavar='s', type=int)
parser.add_argument('--input_current_type', metavar='ic', type=str)

args = parser.parse_args()
print(args)

np.random.seed(args.seed)

SEED = args.seed
POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch
N_INNER_LOOPS = 1 # Number of times to simulate network and plasticity rules per loss function evaluation
STD_EXPL = args.std_expl
DW_LAG = 5
FIXED_DATA = bool(args.fixed_data)
L1_PENALTY = args.l1_pen
R_TARGET = 5
INPUT_CURRENT_TYPE = args.input_current_type

T = 0.2 # Total duration of one network simulation
dt = 1e-4 # Timestep
t = np.linspace(0, T, int(T / dt))
train_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)
test_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)

rule_names = [ # Define labels for all rules to be run during simulations
	r'',
	r'$y$',
	r'$x$',
	# r'$y^2$',
	# r'$x^2$',
	r'$x \, y$',
	# r'$x \, y^2$',
	# r'$x^2 \, y$',
	# r'$x^2 \, y^2$',
	# r'$y_{int}$',
	# r'$x \, y_{int}$',
	# r'$x_{int}$',
	# r'$x_{int} \, y$',
	# r'$x_{int} \, y^2$',
	# r'$y_{int} \, y$',

	# r'$w$',
	# r'$w \, y$',
	# r'$w \, x$',
	# r'$w \, y^2$',
	# # r'$w \, x^2$',
	# r'$w \, x \, y$',
	# r'$w \, x \, y^2$',
	# r'$w \, x^2 \, y$',
	# # r'$w \, x^2 \, y^2$',
	# r'$w y_{int}$',
	# r'$w x \, y_{int}$',
	# # r'$w x_{int}$',
	# r'$w x_{int} \, y$',
	# r'$w x_{int} \, y^2$',
	# r'$w y_{int} \, y$',

	# r'$w^2$',
	# r'$w^2 \, y$',
	# r'$w^2 \, x$',
	# r'$w^2 \, y^2$',
	# r'$w^2 \, x^2$',
	# r'$w^2 \, x \, y$',
	# r'$w^2 \, x \, y^2$',
	# r'$w^2 \, x^2 \, y$',
	# r'$w^2 \, x^2 \, y^2$',
	# r'$w^2 y_{int}$',
	# r'$w^2 x \, y_{int}$',
	# r'$w^2 x_{int}$',
	# r'$w^2 x_{int} \, y$',
]

rule_names = [
	[r'$E \rightarrow E$ ' + r_name for r_name in rule_names],
	# [r'$E \rightarrow I$ ' + r_name for r_name in rule_names],
	# [r'$I \rightarrow E$ ' + r_name for r_name in rule_names],
]
rule_names = np.array(rule_names).flatten()


# Make directory for outputting simulations
if not os.path.exists('sims_out'):
	os.mkdir('sims_out')

# Make subdirectory for this particular experiment
time_stamp = str(datetime.now()).replace(' ', '_')
out_dir = f'sims_out/firing_rate_learning_{BATCH_SIZE}_STD_EXPL_{STD_EXPL}_FIXED_{FIXED_DATA}_L1_PENALTY_{L1_PENALTY}_ICTYPE_{INPUT_CURRENT_TYPE}_SEED_{SEED}_{time_stamp}'
os.mkdir(out_dir)

# Make subdirectory for outputting CMAES info
os.mkdir(os.path.join(out_dir, 'outcmaes'))

# Made CSVs for outputting train & test data
header = ['evals', 'loss'] + [f'true_loss_{i}' for i in np.arange(BATCH_SIZE)]
header += list(rule_names)
header += ['effect_means']
header += ['effect_stds']

train_data_path = os.path.join(out_dir, 'train_data.csv')
write_csv(train_data_path, header)

test_data_path = os.path.join(out_dir, 'test_data.csv')
write_csv(test_data_path, header)


w_e_e = 0.8e-5 / dt

def make_network():
	'''
	Generates an excitatory chain with recurrent inhibition and weak recurrent excitation. Weights that form sequence are distored randomly.

	'''
	w_initial = np.zeros((2, 2))
	w_initial[1, 0] = w_e_e

	return w_initial


def calc_loss(r : np.ndarray):

	if np.isnan(r).any():
		return 1e8

	return np.sum(np.square(r[int(len(t) / 2):, 1] - R_TARGET))


def plot_results(results, eval_tracker, out_dir, plasticity_coefs, true_losses, syn_effect_penalties, train=True):
	scale = 3
	n_res_to_show = BATCH_SIZE

	gs = gridspec.GridSpec(2 * n_res_to_show + 2, 2)
	fig = plt.figure(figsize=(4  * scale, (2 * n_res_to_show + 3) * scale), tight_layout=True)
	axs = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(2 * n_res_to_show)]
	axs += [fig.add_subplot(gs[2 * n_res_to_show, :])]
	axs += [fig.add_subplot(gs[2 * n_res_to_show + 1, :])]

	all_effects = []

	for i in np.arange(BATCH_SIZE):
		# for each network in the batch, graph its excitatory, inhibitory activity, as well as the target activity
		res = results[i]
		r = res['r']
		w = res['w']
		w_initial = res['w_initial']
		effects = res['syn_effects']

		all_effects.append(effects)

		axs[2 * i][0].plot(t, r[:, 0], c='black') # graph excitatory neuron activity
		axs[2 * i][0].plot(t, r[:, 1], c='red') # graph excitatory neuron activity

		vmin = np.min([w_initial.min(), w.min()])
		vmax = np.max([w_initial.max(), w.max()])

		vbound = np.maximum(vmax, np.abs(vmin))

		mappable = axs[2 * i + 1][0].matshow(w_initial, vmin=-vbound, vmax=vbound, cmap='coolwarm') # plot initial weight matrix
		plt.colorbar(mappable, ax=axs[2 * i + 1][0])

		mappable = axs[2 * i + 1][1].matshow(w, vmin=-vbound, vmax=vbound, cmap='coolwarm') # plot final weight matrix
		plt.colorbar(mappable, ax=axs[2 * i + 1][1])

		axs[2 * i][0].set_title(f'{true_losses[i]} + {syn_effect_penalties[i]}')
		for i_axs in range(2):
			# axs[2 * i][i_axs].set_ylim(0, 10.5)
			axs[2 * i][i_axs].set_xlabel('Time (s)')
			axs[2 * i][i_axs].set_ylabel('Firing rate')

	partial_rules_len = int(len(plasticity_coefs))

	all_effects = np.array(all_effects)
	effects = np.mean(all_effects, axis=0)

	axs[2 * n_res_to_show + 1].set_xticks(np.arange(len(effects)))
	effects_argsort = []
	for l in range(1):
		effects_partial = effects[l * partial_rules_len: (l+1) * partial_rules_len]
		effects_argsort_partial = np.flip(np.argsort(effects_partial))
		effects_argsort.append(effects_argsort_partial + l * partial_rules_len)
		x = np.arange(len(effects_argsort_partial)) + l * partial_rules_len
		axs[2 * n_res_to_show + 1].bar(x, effects_partial[effects_argsort_partial], zorder=0)
		for i_e in x:
			axs[2 * n_res_to_show + 1].scatter(i_e * np.ones(all_effects.shape[0]), all_effects[:, effects_argsort_partial][:, i_e], c='black', zorder=1, s=3)
	axs[2 * n_res_to_show + 1].set_xticklabels(rule_names[np.concatenate(effects_argsort)], rotation=60, ha='right')
	axs[2 * n_res_to_show + 1].set_xlim(-1, len(effects))

	true_loss = np.sum(true_losses)
	syn_effect_penalty = np.sum(syn_effect_penalties)
	axs[2 * n_res_to_show].set_title(f'Loss: {true_loss + syn_effect_penalty}, {true_loss}, {syn_effect_penalty}')

	# plot the coefficients assigned to each plasticity rule (unsorted by size)
	for l in range(1):
		axs[2 * n_res_to_show].bar(np.arange(partial_rules_len) + l * partial_rules_len, plasticity_coefs[l * partial_rules_len: (l+1) * partial_rules_len])
	axs[2 * n_res_to_show].set_xticks(np.arange(len(plasticity_coefs)))
	axs[2 * n_res_to_show].set_xticklabels(rule_names, rotation=60, ha='right')
	axs[2 * n_res_to_show].set_xlim(-1, len(plasticity_coefs))

	pad = 4 - len(str(eval_tracker['evals']))
	zero_padding = '0' * pad
	evals = eval_tracker['evals']

	fig.tight_layout()
	if train:
		fig.savefig(f'{out_dir}/{zero_padding}{evals}.png')
	else:
		fig.savefig(f'{out_dir}/{zero_padding}{evals}_test.png')
	plt.close('all')


def calc_alpha_func(tau_alpha):
	alpha_func_n_steps = int(10 * tau_alpha / dt)
	t_alpha = np.arange(0, alpha_func_n_steps) * dt
	return np.e * t_alpha / tau_alpha * np.exp(-t_alpha/tau_alpha)


def poisson_arrivals_to_inputs(arrivals, tau_alpha):
	alpha_func = calc_alpha_func(tau_alpha)
	input_current = np.zeros(arrivals.shape)

	for i in range(arrivals.shape[1]):
		input_current[:, i] = np.convolve(alpha_func, arrivals[:, i], mode='full')[:arrivals.shape[0]]
	return input_current


def simulate_single_network(index, x, track_params=True, train=True):
	'''
	Simulate one set of plasticity rules. `index` describes the simulation's position in the current batch and is used to randomize the random seed.
	'''
	plasticity_coefs = x

	if FIXED_DATA:
		if train:
			np.random.seed(train_seeds[index])
		else:
			np.random.seed(test_seeds[index])
	else:
		np.random.seed()

	w_initial = make_network()

	w = copy(w_initial)
	w_plastic = np.where(w != 0, 1, 0).astype(int) # define non-zero weights as mutable under the plasticity rules

	cumulative_loss = 0

	all_effects = np.zeros(plasticity_coefs.shape)
	all_mean_active_time_diffs = []
	total_activity_loss = 0

	blew_up = False

	surviving_synapse_mask = np.ones((2, 2)).astype(bool)

	for i in range(N_INNER_LOOPS):
		# Define input for activation of the network
		r_in = np.zeros((len(t), 2))
		random_inputs_poisson = np.random.poisson(lam=500 * dt, size=(len(t), 2))
		random_inputs = poisson_arrivals_to_inputs(random_inputs_poisson, 3e-3)

		if INPUT_CURRENT_TYPE == 'HIGH':
			const_input_current = 0.45
		elif INPUT_CURRENT_TYPE == 'LOW':
			const_input_current = 0.15
		elif INPUT_CURRENT_TYPE == 'MIXED':
			if np.random.rand() > 0.5:
				const_input_current = 0.45
			else:
				const_input_current = 0.15

		random_inputs[:, 0] = const_input_current + 0.005 * random_inputs[:, 0] # 0.15 and 0.45 for CONST_INPUT_CURRENT
		random_inputs[:, 1] = 0.005 * random_inputs[:, 1]
		r_in += random_inputs

		# below, simulate one activation of the network for the period T
		r, s, v, w_out, effects = simulate(t, 2, 0, r_in, plasticity_coefs, w, w_plastic, dt=dt, tau_e=10e-3, tau_i=0.1e-3, g=10, w_u=1, track_params=track_params)

		if np.isnan(r).any() or (np.abs(w_out) > 100).any() or (np.abs(w_out) < 1.5e-6).all(): # if simulation turns up nans in firing rate matrix, end the simulation
			return {
				'blew_up': True,
			}
			

		loss = calc_loss(r)
		cumulative_loss += loss

		# all_weight_deltas.append(np.sum(np.abs(w_out - w_hist[0])))

		# w_hist.append(w_out)
		# if len(w_hist) > DW_LAG:
		# 	w_hist.pop(0)

		if effects is not None:
			all_effects += effects

		w = w_out # use output weights evolved under plasticity rules to begin the next simulation

	normed_loss = cumulative_loss

	return {
		'loss': normed_loss,
		'blew_up': False,
		'r': r,
		'w': w,
		'w_initial': w_initial,
		'syn_effects': all_effects,
	}


def log_sim_results(write_path, eval_tracker, loss, true_losses, plasticity_coefs, syn_effects):
	# eval_num, loss, true_losses, plastic_coefs, syn_effects
	syn_effect_means = np.mean(syn_effects, axis=0)
	syn_effect_stds = np.std(syn_effects, axis=0)
	to_save = np.concatenate([[eval_tracker['evals'], loss], true_losses, plasticity_coefs, syn_effect_means, syn_effect_stds]).flatten()
	print(to_save)
	write_csv(write_path, list(to_save))


def process_plasticity_rule_results(results, x, eval_tracker=None, train=True):
	plasticity_coefs = x

	if np.any(np.array([res['blew_up'] for res in results])):
		if eval_tracker is not None:
			eval_tracker['evals'] += 1
		return 1e8 * BATCH_SIZE + 1e7 * np.sum(np.abs(plasticity_coefs)), 1e8 * np.ones((len(results),)), np.zeros((len(results), len(plasticity_coefs)))

	true_losses = np.array([res['loss'] for res in results])
	syn_effects = np.stack([res['syn_effects'] for res in results])
	syn_effect_penalties = np.zeros(syn_effects.shape[0])
	one_third_len = int(syn_effects.shape[1] / 3)

	# for i in range(3):
	# 	syn_effect_penalties += L1_PENALTIES[i] * np.sum(np.abs(syn_effects[:, i * one_third_len:(i+1) * one_third_len]), axis=1)

	syn_effect_penalties += L1_PENALTY * np.sum(np.abs(plasticity_coefs))

	losses = true_losses + syn_effect_penalties
	loss = np.sum(losses)

	if eval_tracker is not None:
		if train:
			if np.isnan(eval_tracker['best_loss']) or loss < eval_tracker['best_loss']:
				if eval_tracker['evals'] > 0:
					eval_tracker['best_loss'] = loss
					eval_tracker['best_changed'] = True
					eval_tracker['plasticity_coefs'] = copy(plasticity_coefs)
				plot_results(results, eval_tracker, out_dir, plasticity_coefs, true_losses, syn_effect_penalties, train=True)
			eval_tracker['evals'] += 1
		else:
			plot_results(results, eval_tracker, out_dir, eval_tracker['plasticity_coefs'], true_losses, syn_effect_penalties, train=False)

	print('guess:', plasticity_coefs)
	print('loss:', loss)
	print('')
	return loss, true_losses, syn_effects


def load_best_params(file_name):
	file_path = f'./sims_out/{file_name}/outcmaes/xrecentbest.dat'
	df_params = read_csv(file_path, read_header=False)
	x = np.arange(df_params.shape[0])
	min_loss_idx = np.argmin([df_params.iloc[i][4] for i in x])
	best_params = df_params.iloc[min_loss_idx][5:]
	return np.array(best_params)


def simulate_single_network_wrapper(tup):
	return simulate_single_network(*tup)


def eval_all(X, eval_tracker=None):
	start = time.time()

	indices = np.arange(BATCH_SIZE)
	pool = mp.Pool(POOL_SIZE)

	task_vars = []
	for x in X:
		for idx in indices:
			task_vars.append((idx, x))
	results = pool.map(simulate_single_network_wrapper, task_vars)

	pool.close()
	pool.join()

	losses = []
	for i in range(len(X)):
		loss, true_losses, syn_effects = process_plasticity_rule_results(results[BATCH_SIZE * i: BATCH_SIZE * (i+1)], X[i], eval_tracker=eval_tracker)
		losses.append(loss)
		log_sim_results(train_data_path, eval_tracker, loss, true_losses, X[i], syn_effects)
	
	dur = time.time() - start
	print('dur:', dur)

	return losses

def process_params_str(s):
	params = []
	for x in s.split(' '):
		x = x.replace('\n', '')
		if x != '':
			params.append(float(x))
	return np.array(params)

if __name__ == '__main__':
	mp.set_start_method('fork')

	if args.load_initial is not None:
		x0 = load_best_params(args.load_initial)
	else:
		x0 = np.zeros(4)

# 	x1 = '''0.03319516 -0.02460002 -0.0194806  -0.031211    0.01034758 -0.01734708
 #  0.01798712  0.00962709  0.00431229  0.04313792  0.04115364 -0.05190074
 #  0.01912274  0.03640839  0.01504871 -0.02156684  0.02122507  0.00741272
 # -0.01294517 -0.00479894  0.00765137  0.02295402 -0.01261857 -0.06408623
 #  0.02498256  0.00175581  0.01820646  0.00683108  0.01163846'''

	# x1 = process_params_str(x1)

	# x0[0] = 0.5 * x0[0]
	# x0[12] = 0.5 * x0[12]

	print(x0)

	eval_tracker = {
		'evals': 0,
		'best_loss': np.nan,
		'best_changed': False,
	}

	eval_all([x0], eval_tracker=eval_tracker)

	options = {
		'verb_filenameprefix': os.path.join(out_dir, 'outcmaes/'),
		'popsize': 8,
	}

	for k in range(10):
		es = cma.CMAEvolutionStrategy(x0, STD_EXPL, options)

		print(es.opts)

		while not es.stop():
			X = es.ask()
			es.tell(X, eval_all(X, eval_tracker=eval_tracker))
			es.disp()

		options['popsize'] += 2
