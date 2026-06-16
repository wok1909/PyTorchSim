import sys, torch, importlib.util
spec = importlib.util.spec_from_file_location(
    "tgp", "/workspace/PyTorchSim/tests/test_gqa_prefill.py")
tgp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tgp)

cfg = tgp.MODEL_CONFIGS["LLAMA4_TP8"]
ok = True
for S in (512, 2048, 1536):
    q, k, v, scale = tgp._make_inputs(cfg, S, torch.float16)
    net = tgp.GQAPrefillFlash(512, 512, True).eval()
    o = net.cpu()(q, k, v, scale=scale)
    r = tgp._reference(q, k, v, scale, True)
    d = (o.float() - r.float()).abs().max().item()
    verdict = "PASS" if d < 0.05 else "FAIL"
    ok = ok and d < 0.05
    print("S=%d shape=%s max|pad_net-sdpa|=%.6f %s" % (S, tuple(o.shape), d, verdict))
print("ALL PASS" if ok else "SOME FAILED")
