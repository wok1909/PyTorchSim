# Minimal STOCK gem5 SE config: run a static RISC-V binary on a stock CPU with a
# settable RVV VLEN. Pure gem5 (no PyTorchSim custom CPU/FU). L1 caches so a small
# array stays cache-resident -> the loop is VPU-compute-bound, not memory-bound.
import argparse, sys, m5
from m5.objects import *
# Make PyTorchSim's custom RiscvVPU available too (subclass of stock RiscvMinorCPU).
sys.path.insert(0, "/workspace/PyTorchSim")
from gem5_script.vpu_config import *
SystolicArray.systolicArrayWidth = 256   # idle for pure-RVV binary, but set to avoid defaults
SystolicArray.systolicArrayHeight = 256

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cmd", required=True)
ap.add_argument("--vlen", type=int, default=256)
ap.add_argument("--cpu", default="RiscvMinorCPU")
args = ap.parse_args()

CPU = {"RiscvMinorCPU": RiscvMinorCPU, "RiscvO3CPU": RiscvO3CPU,
       "RiscvTimingSimpleCPU": RiscvTimingSimpleCPU, "RiscvVPU": RiscvVPU}[args.cpu]

class L1(Cache):
    assoc = 8; tag_latency = 2; data_latency = 2; response_latency = 2
    mshrs = 16; tgts_per_mshr = 20
class L1I(L1): size = "32kB"
class L1D(L1): size = "64kB"   # > 4KB array -> stays resident (compute-bound)

system = System()
system.clk_domain = SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]

system.cpu = CPU()
system.cpu.ArchISA.vlen = args.vlen   # <-- the knob under test (RVV VLEN bits)

system.cpu.icache = L1I(); system.cpu.dcache = L1D()
system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port
system.membus = SystemXBar()
system.cpu.icache.mem_side = system.membus.cpu_side_ports
system.cpu.dcache.mem_side = system.membus.cpu_side_ports
system.cpu.createInterruptController()

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8(range=system.mem_ranges[0])
system.mem_ctrl.port = system.membus.mem_side_ports
system.system_port = system.membus.cpu_side_ports

proc = Process(); proc.cmd = [args.cmd]
system.cpu.workload = proc
system.cpu.createThreads()
system.workload = SEWorkload.init_compatible(args.cmd)

root = Root(full_system=False, system=system)
m5.instantiate()
ev = m5.simulate()
print("VLEN=%d CPU=%s EXIT_CAUSE=%s TICKS=%d" % (args.vlen, args.cpu, ev.getCause(), m5.curTick()))
