# CMEM (SRAM buffer) plan — same policy as tpuv4/gemm_plan.py:
# pin only the first operand (arg0_1) into common memory, stream the rest.
plan = {
    "arg0_1"
}
