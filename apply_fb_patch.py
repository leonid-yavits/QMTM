"""QM-aware feedback flags on top of apply_patch.py (field qm_fb, default 0 = unchanged).
  bit 1 (value 1), Type Ib targeted: when a target-polarity clause does NOT fire, forget
        (decrement, prob 1/s) only literals that are 0 on the sample; satisfied literals kept.
        The over-literal-budget case (clause fired but n_inc > L) keeps the standard Ib.
  bit 2 (value 2), Type II exclude-satisfied: when an opposite-polarity clause fires, also
        decrement (prob 1/s) its included literals that are 1 on the sample, in addition to
        the standard increments of excluded 0-literals.
"""
import sys, pathlib
d = pathlib.Path(sys.argv[1])
h, c = d / "ConvolutionalTsetlinMachine.h", d / "ConvolutionalTsetlinMachine.c"
mh, mc = d / "MultiClassConvolutionalTsetlinMachine.h", d / "MultiClassConvolutionalTsetlinMachine.c"

def sub(p, old, new, count=1):
    s = p.read_text(); n = s.count(old)
    if n != count:
        raise SystemExit(f"{p.name}: expected {count}, found {n}:\n{old}")
    p.write_text(s.replace(old, new))

sub(h, "\tint *n_inc;\n};", "\tint *n_inc;\n\tint qm_fb;\n};")
sub(h, "void qm_seed(unsigned long long seed);",
    "void qm_seed(unsigned long long seed);\n\nvoid tm_set_qm_fb(struct TsetlinMachine *tm, int fb);")
sub(c, "\ttm->qm_abs_tol = 0;\n\ttm->n_inc", "\ttm->qm_abs_tol = 0;\n\ttm->qm_fb = 0;\n\ttm->n_inc")
sub(c, "void qm_seed(unsigned long long seed)\n{",
    "void tm_set_qm_fb(struct TsetlinMachine *tm, int fb)\n{\n\ttm->qm_fb = fb;\n}\n\nvoid qm_seed(unsigned long long seed)\n{")

# Type II: exclude satisfied included literals (first, non-regression occurrence only)
old2 = ("\t\t\t\t\ttm_inc(tm, j, k, (~tm->drop_literal[k]) & (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));\n\t\t\t\t}\n"
        "\t\t\t\tif (tm->qm_mode != 0) tm->n_inc[j] = tm_number_of_include_actions(tm, j);\n")
new2 = ("\t\t\t\t\ttm_inc(tm, j, k, (~tm->drop_literal[k]) & (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));\n\t\t\t\t}\n"
        "\t\t\t\tif (tm->qm_fb & 2) {\n"
        "\t\t\t\t\ttm_initialize_random_streams(tm, j);\n"
        "\t\t\t\t\tfor (int k = 0; k < tm->number_of_ta_chunks; ++k) {\n"
        "\t\t\t\t\t\tint patch = tm->clause_patch[j];\n"
        "\t\t\t\t\t\tunsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;\n"
        "\t\t\t\t\t\ttm_dec(tm, j, k, (~tm->drop_literal[k]) & Xi[patch*tm->number_of_ta_chunks + k] & ta_state[pos] & tm->feedback_to_la[k]);\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t}\n"
        "\t\t\t\tif (tm->qm_mode != 0) tm->n_inc[j] = tm_number_of_include_actions(tm, j);\n")
sub(c, old2, new2)

# Type Ib: targeted forgetting when the clause did not fire
oldb = ("\t\t\t\tfor (int k = 0; k < tm->number_of_ta_chunks; ++k) {\n"
        "\t\t\t\t\ttm_dec(tm, j, k, (~tm->drop_literal[k]) & tm->feedback_to_la[k]);\n\t\t\t\t}\n"
        "\t\t\t\tif (tm->qm_mode != 0) tm->n_inc[j] = tm_number_of_include_actions(tm, j);\n")
newb = ("\t\t\t\tint fired = (tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0;\n"
        "\t\t\t\tfor (int k = 0; k < tm->number_of_ta_chunks; ++k) {\n"
        "\t\t\t\t\tunsigned int m = (~tm->drop_literal[k]) & tm->feedback_to_la[k];\n"
        "\t\t\t\t\tif ((tm->qm_fb & 1) && !fired) m &= ~Xi[k];\n"
        "\t\t\t\t\ttm_dec(tm, j, k, m);\n\t\t\t\t}\n"
        "\t\t\t\tif (tm->qm_mode != 0) tm->n_inc[j] = tm_number_of_include_actions(tm, j);\n")
sub(c, oldb, newb)

sub(mh, "void mc_tm_set_qm(struct MultiClassTsetlinMachine *mc_tm, int mode, int theta_pm, int abs_tol);",
    "void mc_tm_set_qm(struct MultiClassTsetlinMachine *mc_tm, int mode, int theta_pm, int abs_tol);\n\n"
    "void mc_tm_set_qm_fb(struct MultiClassTsetlinMachine *mc_tm, int fb);")
s = mc.read_text()
s += ("\nvoid mc_tm_set_qm_fb(struct MultiClassTsetlinMachine *mc_tm, int fb)\n{\n"
      "\tfor (int i = 0; i < mc_tm->number_of_classes; i++) {\n\t\ttm_set_qm_fb(mc_tm->tsetlin_machines[i], fb);\n\t}\n}\n")
mc.write_text(s)
print("fb patch applied")
