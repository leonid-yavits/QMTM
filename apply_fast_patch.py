"""Speed patch for the QM-patched pyTsetlinMachine core (non-indexed path).

(1) Exact include-chunk lists (always on, bit-exact):
    for each clause, the indices of TA chunks whose include word (MSB state bit) is non-zero
    are cached in nz_idx/nz_cnt. Clause evaluation visits only those chunks. A zero include word
    contributes nothing to an AND test, to the all-exclude test, or to a QM violation count, so the
    clause outputs and the RNG call sequence are identical to the full scan. The list of clause j
    is rebuilt after every feedback step that touches clause j, and all lists are rebuilt lazily
    after (re)initialisation or state loading.
(2) Optional per-literal Bernoulli feedback mask (fast_stream=1, default 0 = original):
    the original draws a normal-approximated count of distinct literals by rejection sampling.
    With fast_stream=1 every literal is selected independently with probability p = 1/s:
    p = 1/2 uses one 32-bit random word per chunk; other p use geometric skipping.
    Marginal selection probability is p in both schemes; fast_stream changes the RNG stream.
Usage: python3 apply_fast_patch.py <pyTsetlinMachine dir>"""
import sys, re

D = sys.argv[1]
H = f"{D}/ConvolutionalTsetlinMachine.h"
CF = f"{D}/ConvolutionalTsetlinMachine.c"
MH = f"{D}/MultiClassConvolutionalTsetlinMachine.h"
MC = f"{D}/MultiClassConvolutionalTsetlinMachine.c"


def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, (old[:60], n)
    return s.replace(old, new)


h = open(H).read()
h = sub(h, "\tunsigned int *pos_mask;\n};", "\tunsigned int *pos_mask;\n\tint fast_stream;\n\tunsigned short *nz_idx;\n\tint *nz_cnt;\n\tint nz_ready;\n};")
h = sub(h, "void tm_set_qm_fb(struct TsetlinMachine *tm, int fb);",
        "void tm_set_qm_fb(struct TsetlinMachine *tm, int fb);\n\nvoid tm_set_fast(struct TsetlinMachine *tm, int fast);")
open(H, "w").write(h)

c = open(CF).read()
# allocation
c = sub(c, "\t\ttm->pos_mask[k] = m;\n\t}\n",
        "\t\ttm->pos_mask[k] = m;\n\t}\n\ttm->fast_stream = 0;\n"
        "\tif (number_of_ta_chunks > 65535) { fprintf(stderr, \"too many TA chunks for nz_idx\\n\"); exit(1); }\n"
        "\ttm->nz_idx = (unsigned short *)malloc(sizeof(unsigned short) * number_of_clauses * number_of_ta_chunks);\n"
        "\ttm->nz_cnt = (int *)malloc(sizeof(int) * number_of_clauses);\n\ttm->nz_ready = 0;\n")
c = sub(c, "\tfor (int k = 0; k < tm->number_of_ta_chunks; ++k) {\n\t\ttm->drop_literal[k] = 0;\n\t}\n}",
        "\tfor (int k = 0; k < tm->number_of_ta_chunks; ++k) {\n\t\ttm->drop_literal[k] = 0;\n\t}\n\ttm->nz_ready = 0;\n}")
c = sub(c, "\tfree(tm->n_pos);\n", "\tfree(tm->n_pos);\n\tfree(tm->nz_idx);\n\tfree(tm->nz_cnt);\n")
# helpers, placed before qm_recompute_all
c = sub(c, "static void qm_recompute_all(struct TsetlinMachine *tm)\n", """static inline void nz_recompute(struct TsetlinMachine *tm, int j)
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
""")
# fast random stream
c = sub(c, """	int n = tm->number_of_features;
	double p = 1.0 / (tm->s + 1.0 * clause * (tm->s_range - tm->s) / tm->number_of_clauses);
""", """	int n = tm->number_of_features;
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
""")
# clause evaluation: QM loop over listed chunks
c = sub(c, """	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
		tm->clause_output[j] = 0;
	}

	if (tm->qm_mode != 0) {""", """	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
		tm->clause_output[j] = 0;
	}

	if (!tm->nz_ready) nz_recompute_all(tm);

	if (tm->qm_mode != 0) {""")
c = sub(c, """				for (int k = 0; k < tm->number_of_ta_chunks; k++) {
					unsigned int inc = ta_state[base + k*tm->number_of_state_bits];""",
        """				unsigned short *idx = &tm->nz_idx[j*tm->number_of_ta_chunks];
				for (int q = 0; q < tm->nz_cnt[j]; q++) {
					int k = idx[q];
					unsigned int inc = ta_state[base + k*tm->number_of_state_bits];""")
# clause evaluation: AND loop over listed chunks
old_and = c[c.index("\t\toutput_one_patches_count = 0;\n\t\tfor (int patch = 0; patch < tm->number_of_patches; ++patch) {\n\t\t\tunsigned int output = 1;"):]
old_and = old_and[:old_and.index("\t\t\tif (output) {")]
new_and = """		output_one_patches_count = 0;
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

"""
c = sub(c, old_and, new_and)
# rebuild list after any feedback on clause j (three sites)
c = sub(c, "if (tm->qm_mode != 0) { tm->n_inc[j] = tm_number_of_include_actions(tm, j); tm->n_pos[j] = tm_number_of_pos_includes(tm, j); }",
        "nz_recompute(tm, j); if (tm->qm_mode != 0) { tm->n_inc[j] = tm_number_of_include_actions(tm, j); tm->n_pos[j] = tm_number_of_pos_includes(tm, j); }", count=3)
# state loading
c = sub(c, """				tm->ta_state[pos] = ta_state[pos];
				pos++;
			}
		}
	}
}""", """				tm->ta_state[pos] = ta_state[pos];
				pos++;
			}
		}
	}
	tm->nz_ready = 0;
}""")
# regression path also edits TA states: invalidate per update (unused here, kept correct)
c = sub(c, "void tm_update_regression(struct TsetlinMachine *tm, unsigned int *Xi, int target)\n{",
        "void tm_update_regression(struct TsetlinMachine *tm, unsigned int *Xi, int target)\n{\n\ttm->nz_ready = 0;")
if "#include <stdio.h>" not in c:
    c = "#include <stdio.h>\n" + c
open(CF, "w").write(c)

mh = open(MH).read()
mh = sub(mh, "void mc_tm_set_qm_fb(struct MultiClassTsetlinMachine *mc_tm, int fb);",
         "void mc_tm_set_qm_fb(struct MultiClassTsetlinMachine *mc_tm, int fb);\n\nvoid mc_tm_set_fast(struct MultiClassTsetlinMachine *mc_tm, int fast);")
open(MH, "w").write(mh)
mc = open(MC).read()
i = mc.index("void mc_tm_set_qm_fb(")
blk = mc[i:]
end = blk.index("\n}\n") + 3
mc = mc[:i + end] + """
void mc_tm_set_fast(struct MultiClassTsetlinMachine *mc_tm, int fast)
{
	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		tm_set_fast(mc_tm->tsetlin_machines[i], fast);
	}
}
""" + mc[i + end:]
open(MC, "w").write(mc)
print("patched")
