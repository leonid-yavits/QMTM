"""Polarity-aware QM (qm_mode 3) on top of apply_patch.py + apply_fb_patch.py.
Literal layout with append_negated: literals [0, F) are x_k, [F, 2F) are NOT x_k.
Mode 3: a clause fires iff every included NEGATED literal is satisfied (strict AND) and
the number of violated included POSITIVE literals <= floor((1 - theta) * n_pos_j).
theta = 100% reduces exactly to the classical clause."""
import sys, pathlib
d = pathlib.Path(sys.argv[1])
h, c = d / "ConvolutionalTsetlinMachine.h", d / "ConvolutionalTsetlinMachine.c"

def sub(p, old, new, count=1):
    s = p.read_text(); n = s.count(old)
    if n != count:
        raise SystemExit(f"{p.name}: expected {count}, found {n}:\n{old}")
    p.write_text(s.replace(old, new))

sub(h, "\tint qm_fb;\n};", "\tint qm_fb;\n\tint *n_pos;\n\tunsigned int *pos_mask;\n};")
sub(c, "\ttm->n_inc = (int *)malloc(sizeof(int) * number_of_clauses);",
    "\ttm->n_inc = (int *)malloc(sizeof(int) * number_of_clauses);\n"
    "\ttm->n_pos = (int *)malloc(sizeof(int) * number_of_clauses);\n"
    "\ttm->pos_mask = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);\n"
    "\tfor (int k = 0; k < number_of_ta_chunks; ++k) {\n"
    "\t\tunsigned int m = 0;\n"
    "\t\tfor (int b = 0; b < 32; ++b) { if (k * 32 + b < number_of_features / 2) m |= (1u << b); }\n"
    "\t\ttm->pos_mask[k] = m;\n"
    "\t}")
sub(c, "\t\ttm->n_inc[j] = 0;\n\t}", "\t\ttm->n_inc[j] = 0;\n\t\ttm->n_pos[j] = 0;\n\t}")
sub(c, "\tfree(tm->n_inc);\n}", "\tfree(tm->n_inc);\n\tfree(tm->n_pos);\n\tfree(tm->pos_mask);\n}")

# count helper: positive included literals of clause j
sub(c, "static void qm_recompute_all(struct TsetlinMachine *tm)\n{",
    "static int tm_number_of_pos_includes(struct TsetlinMachine *tm, int clause)\n{\n"
    "\tunsigned int *ta_state = &tm->ta_state[clause*tm->number_of_ta_chunks*tm->number_of_state_bits];\n"
    "\tint n = 0;\n"
    "\tfor (int k = 0; k < tm->number_of_ta_chunks; ++k) {\n"
    "\t\tunsigned int inc = ta_state[k*tm->number_of_state_bits + tm->number_of_state_bits-1];\n"
    "\t\tif (k == tm->number_of_ta_chunks-1) inc &= tm->filter;\n"
    "\t\tn += __builtin_popcount(inc & tm->pos_mask[k]);\n"
    "\t}\n\treturn n;\n}\n\n"
    "static void qm_recompute_all(struct TsetlinMachine *tm)\n{")
sub(c, "\t\ttm->n_inc[j] = tm_number_of_include_actions(tm, j);\n\t}\n}",
    "\t\ttm->n_inc[j] = tm_number_of_include_actions(tm, j);\n\t\ttm->n_pos[j] = tm_number_of_pos_includes(tm, j);\n\t}\n}")
# keep n_pos in sync at the three feedback sites
sub(c, "if (tm->qm_mode != 0) tm->n_inc[j] = tm_number_of_include_actions(tm, j);",
    "if (tm->qm_mode != 0) { tm->n_inc[j] = tm_number_of_include_actions(tm, j); tm->n_pos[j] = tm_number_of_pos_includes(tm, j); }",
    count=3)

# evaluation: tolerance and violation counting
sub(c, "\t\t\tint tol;\n\t\t\tif (tm->qm_mode == 1) {\n\t\t\t\ttol = ((1000 - tm->qm_theta_pm) * n_inc) / 1000;\n\t\t\t} else {\n\t\t\t\ttol = tm->qm_abs_tol;\n\t\t\t}",
    "\t\t\tint tol;\n\t\t\tif (tm->qm_mode == 1) {\n\t\t\t\ttol = ((1000 - tm->qm_theta_pm) * n_inc) / 1000;\n\t\t\t} else if (tm->qm_mode == 3) {\n"
    "\t\t\t\ttol = ((1000 - tm->qm_theta_pm) * tm->n_pos[j]) / 1000;\n\t\t\t} else {\n\t\t\t\ttol = tm->qm_abs_tol;\n\t\t\t}")
sub(c, "\t\t\t\t\tviol += __builtin_popcount(inc & ~(Xi[patch*tm->number_of_ta_chunks + k] | tm->drop_literal[k]));\n"
       "\t\t\t\t\tif (viol > tol) {\n\t\t\t\t\t\tbreak;\n\t\t\t\t\t}",
    "\t\t\t\t\tunsigned int vb = inc & ~(Xi[patch*tm->number_of_ta_chunks + k] | tm->drop_literal[k]);\n"
    "\t\t\t\t\tif (tm->qm_mode == 3) {\n"
    "\t\t\t\t\t\tif (vb & ~tm->pos_mask[k]) { viol = tol + 1; break; }\n"
    "\t\t\t\t\t\tviol += __builtin_popcount(vb & tm->pos_mask[k]);\n"
    "\t\t\t\t\t} else {\n"
    "\t\t\t\t\t\tviol += __builtin_popcount(vb);\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t\tif (viol > tol) {\n\t\t\t\t\t\tbreak;\n\t\t\t\t\t}")
print("polarity patch applied")
