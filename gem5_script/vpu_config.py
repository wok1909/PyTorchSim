import m5
import os as _os
from m5.objects import *

# Env overrides for SFU (transcendental) calibration sweeps. Defaults preserve
# prior behavior (opLat 10, issueLat default=1).
_SFU_OPLAT = int(_os.environ.get("TORCHSIM_SFU_OPLAT", "10"))
_SFU_ISSUELAT = _os.environ.get("TORCHSIM_SFU_ISSUELAT")

# Systolic-array fill/drain latency. opLat=1 models a 0-depth array (a result in
# 1 cycle), which is unphysical for a DxD weight-stationary array: loading the
# stationary operand + propagating data through the array takes ~array-depth
# cycles. With issueLat=1 the FU stays pipelined (1 matmul/cycle once filled), so
# a full-width tile amortizes the fill while a skinny (decode) tile pays it. This
# is the term the model was missing for single-token decode. Default 1 = prior
# behavior; set TORCHSIM_MXU_OPLAT to the array depth to enable.
_MXU_OPLAT = int(_os.environ.get("TORCHSIM_MXU_OPLAT", "1"))

class SystolicArray(MinorFU):
    unitType = "SystolicArray"
    opClasses = minorMakeOpClassSet(["CustomMatMul", "CustomMatMuliVpush", "CustomMatMulwVpush", "CustomMatMulvpop"])
    opLat = _MXU_OPLAT
    systolicArrayWidth = 128
    systolicArrayHeight = 128

class SparseAccelerator(MinorFU):
    unitType = "SparseAccelerator"
    opClasses = minorMakeOpClassSet(["CustomMatMul", "CustomMatMuliVpush", "CustomMatMulwVpush", "CustomMatMulvpop"])
    opLat = 1

class SpecialFunctionUnit(MinorFU):
    opClasses = minorMakeOpClassSet([
        "CustomVexp",
        "CustomVerf",
        "CustomVtanh",
        "CustomVsin",
        "CustomVcos",
        ])
    opLat = _SFU_OPLAT
    if _SFU_ISSUELAT is not None:
        issueLat = int(_SFU_ISSUELAT)

class MinorFPUnit(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "FloatAdd",
            "FloatCmp",
            "FloatCvt",
            "FloatMult",
            "FloatMultAcc",
            "FloatDiv",
            "FloatMisc",
            "FloatSqrt"
        ]
    )

class MinorVecAdder(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdAdd",
            "SimdFloatAdd",
            "SimdFloatAlu",
            "SimdFloatCmp",
            "SimdShift",
            "SimdShiftAcc",
            "SimdAddAcc",
            "SimdAlu",
            "SimdCmp",
        ]
    )
    opLat = 1

class MinorVecMultiplier(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdMult",
            "SimdFloatMult",
            "SimdMultAcc",
            "SimdMatMultAcc",
            "SimdSqrt",
            "SimdFloatMultAcc",
            "SimdFloatMatMultAcc",
            "SimdFloatSqrt",
        ]
    )
    opLat = 1

class MinorVecDivider(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdDiv",
            "SimdFloatDiv",
        ]
    )
    opLat = 1

class MinorVecReduce(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdReduceAdd",
            "SimdReduceAlu",
            "SimdReduceCmp",
            "SimdFloatReduceAdd",
            "SimdFloatReduceCmp",
        ]
    )
    opLat = 1

class MinorVecLdStore(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdUnitStrideLoad",
            "SimdUnitStrideStore",
            "SimdUnitStrideMaskLoad",
            "SimdUnitStrideMaskStore",
            "SimdStridedLoad",
            "SimdStridedStore",
            "SimdIndexedLoad",
            "SimdIndexedStore",
            "SimdUnitStrideFaultOnlyFirstLoad",
            "SimdWholeRegisterLoad",
            "SimdWholeRegisterStore",
            "SimdUnitStrideSegmentedLoad",
            "SimdUnitStrideSegmentedStore",
        ]
    )
    opLat = 1

class MinorVecMisc(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdCvt",
            "SimdFloatCvt",
            "SimdFloatMisc",
            "SimdPredAlu",
            "SimdMisc",
            "SimdExt",
            "SimdFloatExt",
            "CustomVlaneIdx",
        ]
    )
    opLat = 1

class MinorVecConfig(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdConfig",
        ]
    )
    opLat = 1

class MinorCustomIntFU(MinorDefaultIntFU):
    opLat = 1

class MinorCustomIntDivFU(MinorDefaultIntDivFU):
    opLat = 1

class MinorCustomIntMulFU(MinorDefaultIntMulFU):
    opLat = 1

class MinorCustomPredFU(MinorDefaultPredFU):
    opLat = 1

class MinorCustomMemFU(MinorDefaultMemFU):
    opLat = 1

class MinorCustomMiscFU(MinorDefaultMiscFU):
    opLat = 1

class MinorCustomFUPool(MinorFUPool):
    funcUnits = [
        # Scalar unit
        MinorFPUnit(),
        MinorCustomIntFU(),
        MinorCustomIntFU(),
        MinorCustomIntMulFU(),
        MinorCustomIntDivFU(),
        MinorCustomPredFU(),
        MinorCustomMemFU(),
        MinorCustomMiscFU(),

        # Scalar unit
        MinorFPUnit(),
        MinorCustomIntFU(),
        MinorCustomIntFU(),
        MinorCustomIntMulFU(),
        MinorCustomIntDivFU(),
        MinorCustomPredFU(),
        MinorCustomMemFU(),
        MinorCustomMiscFU(),

        # Matmul unit
        SystolicArray(), # 0
 
        # Vector
        MinorVecConfig(), # 1 for vector config
        MinorVecConfig(),
        MinorVecMisc(),
        MinorVecMisc(),
        MinorVecLdStore(),
        MinorVecLdStore(),

        # Vector ALU0
        MinorVecAdder(), # 6
        MinorVecMultiplier(), # 7
        MinorVecDivider(), # 8
        MinorVecReduce(),

        # Vector ALU1
        MinorVecAdder(), # 18 ~ 29
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecReduce(),

        # Vector
        MinorVecConfig(), # 1 for vector config
        MinorVecConfig(),
        MinorVecMisc(),
        MinorVecMisc(),
        MinorVecLdStore(),
        MinorVecLdStore(),

        # Vector ALU0
        MinorVecAdder(), # 6
        MinorVecMultiplier(), # 7
        MinorVecDivider(), # 8
        MinorVecReduce(),

        # Vector ALU1
        MinorVecAdder(), # 18 ~ 29
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecReduce(),

        # SFU
        SpecialFunctionUnit(),
    ]

class RiscvVPU(RiscvMinorCPU):
    fetch1FetchLimit = 8
    decodeInputWidth = 8
    fetch1ToFetch2BackwardDelay = 0
    fetch2InputBufferSize = 8
    decodeInputBufferSize = 8
    decodeInputWidth = 8
    executeInputBufferSize = 128
    executeInputWidth = 12
    executeIssueLimit = 12
    executeCommitLimit = 12

    # Memory
    executeMemoryIssueLimit = 8
    executeMemoryCommitLimit = 8
    executeMaxAccessesInMemory = 8
    executeLSQMaxStoreBufferStoresPerCycle = 8
    executeLSQTransfersQueueSize = 8
    executeLSQStoreBufferSize = 8

    executeFuncUnits = MinorCustomFUPool()
