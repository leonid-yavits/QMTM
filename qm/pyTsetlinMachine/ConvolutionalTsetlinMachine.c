/*

Copyright (c) 2021 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
https://arxiv.org/abs/1905.09688

*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fast_rand.h"

#include "ConvolutionalTsetlinMachine.h"

struct TsetlinMachine *CreateTsetlinMachine(
	int number_of_clauses,
	int number_of_features,
	int number_of_patches,
	int number_of_ta_chunks,
	int number_of_state_bits,
	int T,
	double s,
	double s_range,
	int boost_true_positive_feedback,
	int weighted_clauses,
	int max_included_literals
)
{
	/* Set up the Tsetlin Machine structure */

	struct TsetlinMachine *tm = (void *)malloc(sizeof(struct TsetlinMachine));

	tm->number_of_clauses = number_of_clauses;

	tm->number_of_features = number_of_features;

	tm->number_of_clause_chunks = (number_of_clauses-1)/32 + 1;

	tm->number_of_patches = number_of_patches;

	tm->clause_output = (unsigned int *)malloc(sizeof(unsigned int) * tm->number_of_clause_chunks);

	tm->drop_clause = (unsigned int *)malloc(sizeof(unsigned int) * tm->number_of_clause_chunks);

	tm->drop_literal = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);

	tm->output_one_patches = (int *)malloc(sizeof(int) * number_of_patches);

	tm->number_of_ta_chunks = number_of_ta_chunks;

	tm->feedback_to_la = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);

	tm->number_of_state_bits = number_of_state_bits;

	tm->ta_state = (unsigned int *)malloc(sizeof(unsigned int) * number_of_clauses * number_of_ta_chunks * number_of_state_bits);

	tm->T = T;

	tm->s = s;

	tm->s_range = s_range;

	tm->clause_patch = (unsigned int *)malloc(sizeof(unsigned int) * number_of_clauses);

	tm->feedback_to_clauses = (int *)malloc(sizeof(int) * tm->number_of_clause_chunks);
	
	tm->clause_weights = (unsigned int *)malloc(sizeof(unsigned int) * number_of_clauses);

	if (((number_of_features) % 32) != 0) {
		tm->filter  = (~(0xffffffff << ((number_of_features) % 32)));
	} else {
		tm->filter = 0xffffffff;
	}

	tm->boost_true_positive_feedback = boost_true_positive_feedback;

	tm->weighted_clauses = weighted_clauses;

	tm->max_included_literals = max_included_literals;

	tm->qm_mode = 0;
	tm->qm_theta_pm = 1000;
	tm->qm_abs_tol = 0;
	tm->qm_fb = 0;
	tm->n_inc = (int *)malloc(sizeof(int) * number_of_clauses);
	tm->n_pos = (int *)malloc(sizeof(int) * number_of_clauses);
	tm->pos_mask = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);
	for (int k = 0; k < number_of_ta_chunks; ++k) {
		unsigned int m = 0;
		for (int b = 0; b < 32; ++b) { if (k * 32 + b < number_of_features / 2) m |= (1u << b); }
		tm->pos_mask[k] = m;
	}
	tm->fast_stream = 0;
	if (number_of_ta_chunks > 65535) { fprintf(stderr, "too many TA chunks for nz_idx\n"); exit(1); }
	tm->nz_idx = (unsigned short *)malloc(sizeof(unsigned short) * number_of_clauses * number_of_ta_chunks);
	tm->nz_cnt = (int *)malloc(sizeof(int) * number_of_clauses);
	tm->nz_ready = 0;
	
	tm_initialize(tm);

	return tm;
}

void tm_initialize(struct TsetlinMachine *tm)
{
	/* Set up the Tsetlin Machine structure */

	unsigned int pos = 0;
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
			for (int b = 0; b < tm->number_of_state_bits-1; ++b) {											
				tm->ta_state[pos] = ~0;
				pos++;
			}
			tm->ta_state[pos] = 0;
			pos++;
		}
		tm->clause_weights[j] = 1;
		tm->n_inc[j] = 0;
		tm->n_pos[j] = 0;
	}

	for (int j = 0; j < tm->number_of_clause_chunks; ++j) {
		tm->drop_clause[j] = 0;
	}

	for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
		tm->drop_literal[k] = 0;
	}
	tm->nz_ready = 0;
}

void tm_destroy(struct TsetlinMachine *tm)
{
	free(tm->clause_output);
	free(tm->drop_clause);
	free(tm->drop_literal);
	free(tm->output_one_patches);
	free(tm->feedback_to_la);
	free(tm->ta_state);
	free(tm->feedback_to_clauses);
	free(tm->clause_weights);
	free(tm->n_inc);
	free(tm->n_pos);
	free(tm->nz_idx);
	free(tm->nz_cnt);
	free(tm->pos_mask);
}

static inline void tm_initialize_random_streams(struct TsetlinMachine *tm, int clause)
{
	// Initialize all bits to zero	
	memset(tm->feedback_to_la, 0, tm->number_of_ta_chunks*sizeof(unsigned int));

	int n = tm->number_of_features;
	double p = 1.0 / (tm->s + 1.0 * clause * (tm->s_range - tm->s) / tm->number_of_clauses);

	if (tm->fast_stream) {
		if (p == 0.5) {
			for (int k = 0; k < tm->number_of_ta_chunks; ++k) tm->feedback_to_la[k] = fast_rand();
			tm->feedback_to_la[tm->number_of_ta_chunks-1] &= tm->filter;
		} else if (p >= 1.0) {
			for (int k = 0; k < tm->number_of_ta_chunks; ++k) tm->feedback_to_la[k] = ~0u;
			tm->feedback_to_la[tm->number_of_ta_chunks-1] &= tm->filter;
		} else {
			double lq = log(1.0 - p);
			long f = -1;
			while (1) {
				double u = ((double)fast_rand() + 1.0) / ((double)FAST_RAND_MAX + 2.0);   /* u in (0,1) */
				f += 1 + (long)floor(log(u) / lq);
				if (f >= n) break;
				tm->feedback_to_la[f / 32] |= 1u << (f % 32);
			}
		}
		return;
	}

	int active = normal(n * p, n * p * (1 - p));
	active = active >= n ? n : active;
	active = active < 0 ? 0 : active;
	while (active--) {
		int f = fast_rand() % (tm->number_of_features);
		while (tm->feedback_to_la[f / 32] & (1 << (f % 32))) {
			f = fast_rand() % (tm->number_of_features);
	    }
		tm->feedback_to_la[f / 32] |= 1 << (f % 32);
	}
}

// Counts the number of included literals of a clause
int tm_number_of_include_actions(struct TsetlinMachine *tm, int clause)
{	
	unsigned int clause_pos = clause*tm->number_of_ta_chunks*tm->number_of_state_bits;
	unsigned int *ta_state = &tm->ta_state[clause_pos];

	int number_of_include_actions = 0;
	for (int k = 0; k < tm->number_of_ta_chunks-1; ++k) {
		unsigned int ta_pos = k*tm->number_of_state_bits + tm->number_of_state_bits-1;
		number_of_include_actions += __builtin_popcount(ta_state[ta_pos]);
	}

	unsigned int ta_pos = (tm->number_of_ta_chunks-1)*tm->number_of_state_bits + tm->number_of_state_bits-1;
	number_of_include_actions += __builtin_popcount(ta_state[ta_pos] & tm->filter);

	return(number_of_include_actions);
}

static int tm_number_of_pos_includes(struct TsetlinMachine *tm, int clause)
{
	unsigned int *ta_state = &tm->ta_state[clause*tm->number_of_ta_chunks*tm->number_of_state_bits];
	int n = 0;
	for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
		unsigned int inc = ta_state[k*tm->number_of_state_bits + tm->number_of_state_bits-1];
		if (k == tm->number_of_ta_chunks-1) inc &= tm->filter;
		n += __builtin_popcount(inc & tm->pos_mask[k]);
	}
	return n;
}

static inline void nz_recompute(struct TsetlinMachine *tm, int j)
{
	unsigned int base = j*tm->number_of_ta_chunks*tm->number_of_state_bits + tm->number_of_state_bits-1;
	unsigned short *idx = &tm->nz_idx[j*tm->number_of_ta_chunks];
	int n = 0;
	for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
		unsigned int inc = tm->ta_state[base + k*tm->number_of_state_bits];
		if (k == tm->number_of_ta_chunks-1) inc &= tm->filter;
		if (inc) idx[n++] = (unsigned short)k;
	}
	tm->nz_cnt[j] = n;
}

static void nz_recompute_all(struct TsetlinMachine *tm)
{
	for (int j = 0; j < tm->number_of_clauses; ++j) nz_recompute(tm, j);
	tm->nz_ready = 1;
}

void tm_set_fast(struct TsetlinMachine *tm, int fast)
{
	tm->fast_stream = fast;
}

static void qm_recompute_all(struct TsetlinMachine *tm)
{
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		tm->n_inc[j] = tm_number_of_include_actions(tm, j);
		tm->n_pos[j] = tm_number_of_pos_includes(tm, j);
	}
}

void tm_set_qm(struct TsetlinMachine *tm, int mode, int theta_pm, int abs_tol)
{
	tm->qm_mode = mode;
	tm->qm_theta_pm = theta_pm;
	tm->qm_abs_tol = abs_tol;
	qm_recompute_all(tm);
}

void tm_set_qm_fb(struct TsetlinMachine *tm, int fb)
{
	tm->qm_fb = fb;
}

void qm_seed(unsigned long long seed)
{
	mcg_state = 2u * (uint64_t)seed + 1u;
	pcg32_fast();
	srand((unsigned int)seed);
}

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void tm_inc(struct TsetlinMachine *tm, int clause, int chunk, unsigned int active)
{
	unsigned int carry, carry_next;

	unsigned int *ta_state = &tm->ta_state[clause*tm->number_of_ta_chunks*tm->number_of_state_bits + chunk*tm->number_of_state_bits];

	carry = active;
	for (int b = 0; b < tm->number_of_state_bits; ++b) {
		carry_next = ta_state[b] & carry; // Sets carry bits (overflow) passing on to next bit
		ta_state[b] = ta_state[b] ^ carry; // Performs increments with XOR
		carry = carry_next;
	}

	if (carry > 0) {
		for (int b = 0; b < tm->number_of_state_bits; ++b) {
			ta_state[b] |= carry;
		}
	} 	
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void tm_dec(struct TsetlinMachine *tm, int clause, int chunk, unsigned int active)
{
	unsigned int carry, carry_next;

	unsigned int *ta_state = &tm->ta_state[clause*tm->number_of_ta_chunks*tm->number_of_state_bits + chunk*tm->number_of_state_bits];

	carry = active;
	for (int b = 0; b < tm->number_of_state_bits; ++b) {
		carry_next = (~ta_state[b]) & carry; // Sets carry bits (overflow) passing on to next bit
		ta_state[b] = ta_state[b] ^ carry; // Performs increments with XOR
		carry = carry_next;
	}

	if (carry > 0) {
		for (int b = 0; b < tm->number_of_state_bits; ++b) {
			ta_state[b] &= ~carry;
		}
	} 
}

/* Sum up the votes for each class */
static inline int sum_up_class_votes(struct TsetlinMachine *tm)
{
	int class_sum = 0;

	for (int j = 0; j < tm->number_of_clauses; j++) {
		int clause_chunk = j / 32;
		int clause_pos = j % 32;

		if (j % 2 == 0) {
			class_sum += tm->clause_weights[j] * ((tm->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
		} else {
			class_sum -= tm->clause_weights[j] * ((tm->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
		}	
	}

	class_sum = (class_sum > (tm->T)) ? (tm->T) : class_sum;
	class_sum = (class_sum < -(tm->T)) ? -(tm->T) : class_sum;

	return class_sum;
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
static inline void tm_calculate_clause_output(struct TsetlinMachine *tm, unsigned int *Xi, int predict)
{
	int output_one_patches_count;

	unsigned int *ta_state = tm->ta_state;

	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
		tm->clause_output[j] = 0;
	}

	if (!tm->nz_ready) nz_recompute_all(tm);

	if (tm->qm_mode != 0) {
		for (int j = 0; j < tm->number_of_clauses; j++) {
			unsigned int clause_chunk = j / 32;
			unsigned int clause_chunk_pos = j % 32;

			if ((tm->drop_clause[clause_chunk] & (1 << clause_chunk_pos))) {
				continue;
			}

			int n_inc = tm->n_inc[j];
			int tol;
			if (tm->qm_mode == 1) {
				tol = ((1000 - tm->qm_theta_pm) * n_inc) / 1000;
			} else if (tm->qm_mode == 3) {
				tol = ((1000 - tm->qm_theta_pm) * tm->n_pos[j]) / 1000;
			} else {
				tol = tm->qm_abs_tol;
			}

			output_one_patches_count = 0;
			for (int patch = 0; patch < tm->number_of_patches; ++patch) {
				int viol = 0;
				unsigned int base = j*tm->number_of_ta_chunks*tm->number_of_state_bits + tm->number_of_state_bits-1;
				unsigned short *idx = &tm->nz_idx[j*tm->number_of_ta_chunks];
				for (int q = 0; q < tm->nz_cnt[j]; q++) {
					int k = idx[q];
					unsigned int inc = ta_state[base + k*tm->number_of_state_bits];
					if (k == tm->number_of_ta_chunks-1) {
						inc &= tm->filter;
					}
					unsigned int vb = inc & ~(Xi[patch*tm->number_of_ta_chunks + k] | tm->drop_literal[k]);
					if (tm->qm_mode == 3) {
						if (vb & ~tm->pos_mask[k]) { viol = tol + 1; break; }
						viol += __builtin_popcount(vb & tm->pos_mask[k]);
					} else {
						viol += __builtin_popcount(vb);
					}
					if (viol > tol) {
						break;
					}
				}
				int output = (viol <= tol) && !(predict == PREDICT && n_inc == 0);
				if (output) {
					tm->output_one_patches[output_one_patches_count] = patch;
					output_one_patches_count++;
				}
			}

			if (output_one_patches_count > 0) {
				tm->clause_output[clause_chunk] |= (1 << clause_chunk_pos);
				int patch_id = fast_rand() % output_one_patches_count;
				tm->clause_patch[j] = tm->output_one_patches[patch_id];
			}
		}
		return;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		if ((tm->drop_clause[clause_chunk] & (1 << clause_chunk_pos))) {
			continue;
		}

		output_one_patches_count = 0;
		for (int patch = 0; patch < tm->number_of_patches; ++patch) {
			unsigned int output = 1;
			unsigned int base = j*tm->number_of_ta_chunks*tm->number_of_state_bits + tm->number_of_state_bits-1;
			unsigned short *idx = &tm->nz_idx[j*tm->number_of_ta_chunks];
			for (int q = 0; q < tm->nz_cnt[j]; q++) {
				int k = idx[q];
				unsigned int inc = ta_state[base + k*tm->number_of_state_bits];
				if (k == tm->number_of_ta_chunks-1) inc &= tm->filter;
				if ((inc & (Xi[patch*tm->number_of_ta_chunks + k] | tm->drop_literal[k])) != inc) { output = 0; break; }
			}
			unsigned int all_exclude = (tm->nz_cnt[j] == 0);

			output = output && !(predict == PREDICT && all_exclude == 1);

			if (output) {
				tm->output_one_patches[output_one_patches_count] = patch;
				output_one_patches_count++;
			}
		}
	
		if (output_one_patches_count > 0) {
			unsigned int clause_chunk = j / 32;
			unsigned int clause_chunk_pos = j % 32;

 			tm->clause_output[clause_chunk] |= (1 << clause_chunk_pos);

 			int patch_id = fast_rand() % output_one_patches_count;
	 		tm->clause_patch[j] = tm->output_one_patches[patch_id];
 		}
 	}
}

/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void tm_update_clauses(struct TsetlinMachine *tm, unsigned int *Xi, int class_sum, int target)
{
	unsigned int *ta_state = tm->ta_state;

	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
	 	tm->feedback_to_clauses[j] = 0;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		if ((tm->drop_clause[clause_chunk] & (1 << clause_chunk_pos))) {
			continue;
		}

	 	tm->feedback_to_clauses[clause_chunk] |= (((float)fast_rand())/((float)FAST_RAND_MAX) <= (1.0/(tm->T*2))*(tm->T + (1 - 2*target)*class_sum)) << clause_chunk_pos;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		if (!(tm->feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos))) {
			continue;
		}
		
		if ((2*target-1) * (1 - 2 * (j & 1)) == -1) {
			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type II Feedback
				
				if (tm->weighted_clauses && tm->clause_weights[j] > 1) {
					tm->clause_weights[j]--;
				}

				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;

					tm_inc(tm, j, k, (~tm->drop_literal[k]) & (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));
				}
				if (tm->qm_fb & 2) {
					tm_initialize_random_streams(tm, j);
					for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
						int patch = tm->clause_patch[j];
						unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;
						tm_dec(tm, j, k, (~tm->drop_literal[k]) & Xi[patch*tm->number_of_ta_chunks + k] & ta_state[pos] & tm->feedback_to_la[k]);
					}
				}
				nz_recompute(tm, j); if (tm->qm_mode != 0) { tm->n_inc[j] = tm_number_of_include_actions(tm, j); tm->n_pos[j] = tm_number_of_pos_includes(tm, j); }
			}
		} else if ((2*target-1) * (1 - 2 * (j & 1)) == 1) {
			// Type I Feedback

			tm_initialize_random_streams(tm, j);

			if (((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) && (tm_number_of_include_actions(tm, j) <= tm->max_included_literals)) {
				// Type Ia Feedback

				if (tm->weighted_clauses) {
					tm->clause_weights[j]++;
				}
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					if (tm->boost_true_positive_feedback == 1) {
		 				tm_inc(tm, j, k, (~tm->drop_literal[k]) & Xi[patch*tm->number_of_ta_chunks + k]);
					} else {
						tm_inc(tm, j, k, (~tm->drop_literal[k]) & Xi[patch*tm->number_of_ta_chunks + k] & (~tm->feedback_to_la[k]));
					}
		 			
		 			tm_dec(tm, j, k, (~tm->drop_literal[k]) & (~Xi[patch*tm->number_of_ta_chunks + k]) & tm->feedback_to_la[k]);
				}
				nz_recompute(tm, j); if (tm->qm_mode != 0) { tm->n_inc[j] = tm_number_of_include_actions(tm, j); tm->n_pos[j] = tm_number_of_pos_includes(tm, j); }
			} else {
				// Type Ib Feedback
				
				int fired = (tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0;
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					unsigned int m = (~tm->drop_literal[k]) & tm->feedback_to_la[k];
					if ((tm->qm_fb & 1) && !fired) m &= ~Xi[k];
					tm_dec(tm, j, k, m);
				}
				nz_recompute(tm, j); if (tm->qm_mode != 0) { tm->n_inc[j] = tm_number_of_include_actions(tm, j); tm->n_pos[j] = tm_number_of_pos_includes(tm, j); }
			}
		}
	}
}

void tm_update(struct TsetlinMachine *tm, unsigned int *Xi, int target)
{
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, UPDATE);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	int class_sum = sum_up_class_votes(tm);

	/*********************************/
	/*** Train Individual Automata ***/
	/*********************************/
	
	tm_update_clauses(tm, Xi, class_sum, target);
}

int tm_score(struct TsetlinMachine *tm, unsigned int *Xi) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, PREDICT);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	return sum_up_class_votes(tm);
}

int tm_ta_state(struct TsetlinMachine *tm, int clause, int ta)
{
	int ta_chunk = ta / 32;
	int chunk_pos = ta % 32;

	unsigned int pos = clause * tm->number_of_ta_chunks * tm->number_of_state_bits + ta_chunk * tm->number_of_state_bits;

	int state = 0;
	for (int b = 0; b < tm->number_of_state_bits; ++b) {
		if (tm->ta_state[pos + b] & (1 << chunk_pos)) {
			state |= 1 << b; 
		}
	}

	return state;
}

int tm_ta_action(struct TsetlinMachine *tm, int clause, int ta)
{
	int ta_chunk = ta / 32;
	int chunk_pos = ta % 32;

	unsigned int pos = clause * tm->number_of_ta_chunks * tm->number_of_state_bits + ta_chunk * tm->number_of_state_bits + tm->number_of_state_bits-1;

	return (tm->ta_state[pos] & (1 << chunk_pos)) > 0;
}

/*****************************************************/
/*** Storing and Loading of Tsetlin Machine State ****/
/*****************************************************/

void tm_get_ta_state(struct TsetlinMachine *tm, unsigned int *ta_state)
{
	int pos = 0;
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
			for (int b = 0; b < tm->number_of_state_bits; ++b) {
				ta_state[pos] = tm->ta_state[pos];
				pos++;
			}
		}
	}
}

void tm_set_ta_state(struct TsetlinMachine *tm, unsigned int *ta_state)
{
	int pos = 0;
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
			for (int b = 0; b < tm->number_of_state_bits; ++b) {
				tm->ta_state[pos] = ta_state[pos];
				pos++;
			}
		}
	}
	tm->nz_ready = 0;
}

void tm_get_clause_weights(struct TsetlinMachine *tm, unsigned int *clause_weights)
{
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		clause_weights[j] = tm->clause_weights[j];
	}
}

void tm_set_clause_weights(struct TsetlinMachine *tm, unsigned int *clause_weights)
{
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		tm->clause_weights[j] = clause_weights[j];
	}
}

void tm_get_state(struct TsetlinMachine *tm, unsigned int *clause_weights, unsigned int *ta_state)
{
	tm_get_clause_weights(tm, clause_weights);
	tm_get_ta_state(tm, ta_state);
}

void tm_set_state(struct TsetlinMachine *tm, unsigned int *clause_weights, unsigned int *ta_state)
{
	tm_set_clause_weights(tm, clause_weights);
	tm_set_ta_state(tm, ta_state);
	qm_recompute_all(tm);
}

/**************************************/
/*** The Regression Tsetlin Machine ***/
/**************************************/

/* Sum up the votes for each class */
static inline int sum_up_class_votes_regression(struct TsetlinMachine *tm)
{
	int class_sum = 0;

	for (int j = 0; j < tm->number_of_clauses; j++) {
		int clause_chunk = j / 32;
		int clause_pos = j % 32;

		class_sum += tm->clause_weights[j] * ((tm->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
		
	}
	class_sum = (class_sum > (tm->T)) ? (tm->T) : class_sum;

	return class_sum;
}

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void tm_update_regression(struct TsetlinMachine *tm, unsigned int *Xi, int target)
{
	tm->nz_ready = 0;
	unsigned int *ta_state = tm->ta_state;

	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, UPDATE);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	int class_sum = sum_up_class_votes_regression(tm);

	/*********************************/
	/*** Train Individual Automata ***/
	/*********************************/
	
	// Calculate feedback to clauses

	int prediction_error = class_sum - target; 

	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
	 	tm->feedback_to_clauses[j] = 0;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

	 	tm->feedback_to_clauses[clause_chunk] |= (((float)fast_rand())/((float)FAST_RAND_MAX) <= pow(1.0*prediction_error/tm->T, 2)) << clause_chunk_pos;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		if (!(tm->feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos))) {
			continue;
		}
		
		if (prediction_error > 0) {
			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type II Feedback
				
				if (tm->weighted_clauses && tm->clause_weights[j] > 1) {
					tm->clause_weights[j]--;
				}

				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;

					tm_inc(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));
				}
			}
		} else if (prediction_error < 0) {
			// Type I Feedback

			tm_initialize_random_streams(tm, j);

			if (((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) && (tm_number_of_include_actions(tm, j) <= tm->max_included_literals)) {
				// Type Ia Feedback
				
				if (tm->weighted_clauses) {
					tm->clause_weights[j]++;
				}
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					if (tm->boost_true_positive_feedback == 1) {
		 				tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k]);
					} else {
						tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k] & (~tm->feedback_to_la[k]));
					}
		 			
		 			tm_dec(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & tm->feedback_to_la[k]);
				}
			} else {
				// Type Ib Feedback
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					tm_dec(tm, j, k, tm->feedback_to_la[k]);
				}
			}
		}
	}
}

void tm_fit_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples, int epochs)
{
	unsigned int step_size = tm->number_of_patches * tm->number_of_ta_chunks;

	for (int epoch = 0; epoch < epochs; epoch++) {
		// Add shuffling here...
		unsigned int pos = 0;
		for (int i = 0; i < number_of_examples; i++) {
			tm_update_regression(tm, &X[pos], y[i]);
			pos += step_size;
		}
	}
}

int tm_score_regression(struct TsetlinMachine *tm, unsigned int *Xi) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, PREDICT);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	return sum_up_class_votes_regression(tm);
}

/******************************/
/*** Predict y for inputs X ***/
/******************************/

void tm_predict_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples)
{
	unsigned int step_size = tm->number_of_patches * tm->number_of_ta_chunks;

	unsigned int pos = 0;
	for (int l = 0; l < number_of_examples; l++) {
		// Identify class with largest output
		y[l] = tm_score_regression(tm, &X[pos]);
		
		pos += step_size;
	}
	
	return;
}
