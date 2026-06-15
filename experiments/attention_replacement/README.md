# Attention Replacement Probe

Goal: test whether the matrix backend can replace a frozen `torch.nn.MultiheadAttention` block.

Teacher:

```text
x -> frozen MultiheadAttention -> teacher_out
```

Student:

```text
x -> SequenceEvidenceMatrix -> TaskMatrix -> MatrixBackend -> SequenceHead -> student_out
```

Loss:

```text
MSE(student_out, teacher_out) + cosine alignment
```

Important rule: only input builder and head are task-specific. The core remains matrix-space based.

The first implementation target is `matrix_attention_replacement_probe_v1.py`.
