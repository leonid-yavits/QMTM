"""Apply the QMTM (qualified-majority Tsetlin machine) patch to a copy of
pyTsetlinMachine 0.6.6 (non-indexed path only).

Clause rule. Baseline clause j fires iff every included literal is 1, i.e.,
AND_k (~include_jk | x_k). QMTM counts violations v_j = #{k: include_jk & ~x_k}
and fires iff v_j <= tol_j:
  mode 0: baseline (original code path, untouched)
  mode 1: relative QM, tol_j = floor((1 - theta) * n_inc_j)
          ("at least theta of the included literals are satisfied")
  mode 2: absolute QM, tol_j = abs_tol for every clause
          (equals a theta over all M OR terms: theta = 1 - abs_tol/M)
theta = 1 (mode 1) or abs_tol = 0 (mode 2) reduce exactly to the baseline.
Feedback rules (Type I/II) are unchanged.
"""
import sys, pathlib

d = pathlib.Path(sys.argv[1])
h = d / "ConvolutionalTsetlinMachine.h"
c = d / "ConvolutionalTsetlinMachine.c"
mh = d / "MultiClassConvolutionalTsetlinMachine.h"
mc = d / "MultiClassConvolutionalTsetlinMachine.c"

def sub(path, old, new, count=1):
    s = path.read_text()
    n = s.count(old)
    if n != count:
        raise SystemExit(f"{path.name}: expected {count} match(es), found {n} for:\n{old}")
    path.write_text(s.replace(old, new))

# ---- header ----
sub(h, "\tint max_included_literals;\n};",
    "\tint max_included_literals;\n\n\t/* QMTM */\n\tint qm_mode;\n\tint qm_theta_pm;\n\tint qm_abs_tol;\n\tint *n_inc;\n};")
sub(h, "void tm_initialize(struct TsetlinMachine *tm);",
    "void tm_initialize(struct TsetlinMachine *tm);\n\n"
    "void tm_set_qm(struct TsetlinMachine *tm, int mode, int theta_pm, int abs_tol);\n\n"
    "void qm_seed(unsigned long long seed);\n\n"
    "int tm_number_of_include_actions(struct TsetlinMachine *tm, int clause);")

# ---- create / init / destroy ----
sub(c, "\ttm->max_included_literals = max_included_literals;\n\t\n\ttm_initialize(tm);",
    "\ttm->max_included_literals = max_included_literals;\n\n"
    "\ttm->qm_mode = 0;\n\ttm->qm_theta_pm = 1000;\n\ttm->qm_abs_tol = 0;\n"
    "\ttm->n_inc = (int *)malloc(sizeof(int) * number_of_clauses);\n\t\n\ttm_initialize(tm);")
sub(c, "\t\ttm->clause_weights[j] = 1;\n\t}",
    "\t\ttm->clause_weights[j] = 1;\n\t\ttm->n_inc[j] = 0;\n\t}")
sub(c, "\tfree(tm->clause_weights);\n}",
    "\tfree(tm->clause_weights);\n\tfree(tm->n_inc);\n}")

# ---- setter, seed, recompute helper (placed before tm_inc) ----
helpers = r'''
static void qm_recompute_all(struct TsetlinMachine *tm)
{
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		tm->n_inc[j] = tm_number_of_include_actions(tm, j);
	}
}

void tm_set_qm(struct TsetlinMachine *tm, int mode, int theta_pm, int abs_tol)
{
	tm->qm_mode = mode;
	tm->qm_theta_pm = theta_pm;
	tm->qm_abs_tol = abs_tol;
	qm_recompute_all(tm);
}

void qm_seed(unsigned long long seed)
{
	mcg_state = 2u * (uint64_t)seed + 1u;
	pcg32_fast();
	srand((unsigned int)seed);
}

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.'''
sub(c, "\n// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.", helpers)

# ---- clause output: QM branch in front of the untouched original loop ----
qm_eval = r'''	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
		tm->clause_output[j] = 0;
	}

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
			} else {
				tol = tm->qm_abs_tol;
			}

			output_one_patches_count = 0;
			for (int patch = 0; patch < tm->number_of_patches; ++patch) {
				int viol = 0;
				unsigned int base = j*tm->number_of_ta_chunks*tm->number_of_state_bits + tm->number_of_state_bits-1;
				for (int k = 0; k < tm->number_of_ta_chunks; k++) {
					unsigned int inc = ta_state[base + k*tm->number_of_state_bits];
					if (k == tm->number_of_ta_chunks-1) {
						inc &= tm->filter;
					}
					viol += __builtin_popcount(inc & ~(Xi[patch*tm->number_of_ta_chunks + k] | tm->drop_literal[k]));
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
'''
sub(c, "\tfor (int j = 0; j < tm->number_of_clause_chunks; j++) {\n\t\ttm->clause_output[j] = 0;\n\t}\n", qm_eval)

# ---- keep n_inc in sync after feedback to clause j (only when QM is on) ----
# Type II
sub(c, "\t\t\t\t\ttm_inc(tm, j, k, (~tm->drop_literal[k]) & (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));\n\t\t\t\t}\n",
    "\t\t\t\t\ttm_inc(tm, j, k, (~tm->drop_literal[k]) & (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));\n\t\t\t\t}\n"
    "\t\t\t\tif (tm->qm_mode != 0) tm->n_inc[j] = tm_number_of_include_actions(tm, j);\n")
# Type Ia
sub(c, "\t\t \t\t\ttm_dec(tm, j, k, (~tm->drop_literal[k]) & (~Xi[patch*tm->number_of_ta_chunks + k]) & tm->feedback_to_la[k]);\n\t\t\t\t}\n",
    "\t\t \t\t\ttm_dec(tm, j, k, (~tm->drop_literal[k]) & (~Xi[patch*tm->number_of_ta_chunks + k]) & tm->feedback_to_la[k]);\n\t\t\t\t}\n"
    "\t\t\t\tif (tm->qm_mode != 0) tm->n_inc[j] = tm_number_of_include_actions(tm, j);\n")
# Type Ib
sub(c, "\t\t\t\t\ttm_dec(tm, j, k, (~tm->drop_literal[k]) & tm->feedback_to_la[k]);\n\t\t\t\t}\n",
    "\t\t\t\t\ttm_dec(tm, j, k, (~tm->drop_literal[k]) & tm->feedback_to_la[k]);\n\t\t\t\t}\n"
    "\t\t\t\tif (tm->qm_mode != 0) tm->n_inc[j] = tm_number_of_include_actions(tm, j);\n")

# state restore keeps cache valid
sub(c, "\ttm_set_clause_weights(tm, clause_weights);\n\ttm_set_ta_state(tm, ta_state);\n}",
    "\ttm_set_clause_weights(tm, clause_weights);\n\ttm_set_ta_state(tm, ta_state);\n\tqm_recompute_all(tm);\n}")

# ---- multiclass setter ----
sub(mh, "void mc_tm_initialize(struct MultiClassTsetlinMachine *mc_tm);",
    "void mc_tm_initialize(struct MultiClassTsetlinMachine *mc_tm);\n\n"
    "void mc_tm_set_qm(struct MultiClassTsetlinMachine *mc_tm, int mode, int theta_pm, int abs_tol);")
s = mc.read_text()
s += r'''

void mc_tm_set_qm(struct MultiClassTsetlinMachine *mc_tm, int mode, int theta_pm, int abs_tol)
{
	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		tm_set_qm(mc_tm->tsetlin_machines[i], mode, theta_pm, abs_tol);
	}
}
'''
mc.write_text(s)
print("patched", d)
