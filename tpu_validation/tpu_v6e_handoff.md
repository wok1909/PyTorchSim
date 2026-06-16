# TPU v6e (Trillium) 임대 + 프로파일링 핸드오프

이 문서는 Google Cloud에서 TPU v6e를 빌려서 워크로드를 돌리고 TensorBoard 프로파일을 받아오는 전체 과정을 정리한 것입니다.

## 0. 컨텍스트 (이미 검증됨)

- **계정**: `okkyun.w@gmail.com` (gcloud 활성)
- **프로젝트**: `tpu-board` (실제 PROJECT_ID, 디스플레이 이름은 "TPU-board")
- **빌링**: `010005-50DF18-C442F4` (My Billing Account 1, 활성)
- **TPU v6e quota** (이미 부여됨):
  - On-demand: 512 코어
  - Preemptible: 1,536 코어
  - 적용 zone: europe-west4-{a,b,c}, asia-northeast1-{a,b,c}, us-east5-{a,b,c}, us-south1-{a,b,c}, us-central1-{a,b,c,f} 등 다수
- **검증된 동작 zone**: `europe-west4-a` (성공)
- **검증 실패 zone**: `asia-northeast1-b` (zone-level capacity 부족으로 "Failed to perform tenant project creation" 에러 — 도쿄가 가깝지만 capacity 없는 시점 있음. EU로 우회 가능)

---

## 1. 사전 준비 (로컬 머신)

### 1.1 gcloud SDK 설치

macOS:
```bash
brew install --cask google-cloud-sdk
# 또는 공식: https://cloud.google.com/sdk/docs/install
```

확인:
```bash
gcloud --version  # 548.0.0 이상 권장
```

### 1.2 인증 + 프로젝트 설정

```bash
gcloud auth login
gcloud auth application-default login   # ADC (JAX/Python SDK용)
gcloud config set project tpu-board
gcloud config set account okkyun.w@gmail.com
```

확인:
```bash
gcloud config get-value project   # → tpu-board
gcloud auth list                  # → okkyun.w@gmail.com에 *(active) 표시
```

### 1.3 (한 번만) TPU API + service identity 활성화

이미 한 번 했으면 skip해도 됨. 새 프로젝트면 필수:
```bash
gcloud services enable tpu.googleapis.com compute.googleapis.com --project=tpu-board

# Service identity 생성 — gcloud beta 컴포넌트 없으면 REST API로:
TOKEN=$(gcloud auth print-access-token)
curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "https://serviceusage.googleapis.com/v1beta1/projects/tpu-board/services/tpu.googleapis.com:generateServiceIdentity"
# 응답에 "service-138690841234@cloud-tpu.iam.gserviceaccount.com" 같은 이메일 나오면 성공
```

---

## 2. TPU v6e 가용성 확인

### 2.1 Quota 조회 (REST)

```bash
TOKEN=$(gcloud auth print-access-token)
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://cloudquotas.googleapis.com/v1/projects/tpu-board/locations/global/services/tpu.googleapis.com/quotaInfos?pageSize=300" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for q in data.get('quotaInfos', []):
    if 'V6E' in q.get('metricDisplayName','').upper():
        print(q['metricDisplayName'])
        for d in q.get('dimensionsInfos', []):
            v = d.get('details',{}).get('value','0')
            if v != '0': print(f'  {d[\"applicableLocations\"]}: {v}')
"
```

### 2.2 하드웨어 가용 zone 확인

```bash
for zone in europe-west4-a europe-west4-b asia-northeast1-b us-east5-a us-east5-b us-east5-c us-south1-a; do
  has=$(gcloud compute tpus accelerator-types list --zone=$zone 2>/dev/null | grep -c "^v6e-")
  [ "$has" -gt 0 ] && echo "$zone: $(gcloud compute tpus accelerator-types list --zone=$zone | grep "^v6e-" | sort -V | tr '\n' ' ')"
done
```

기대값: 위 zone들 모두 `v6e-1 v6e-4 v6e-8 v6e-16 v6e-32 v6e-64 v6e-128 v6e-256` 노출.

---

## 3. TPU 생성

### 3.1 기본 옵션 (검증된 조합)

```bash
TPU_NAME=okkyun-v6e-test
ZONE=europe-west4-a            # 도쿄 실패시 EU 권장 (검증됨)
ACCEL=v6e-1                    # 가장 작음 = 1 chip
RUNTIME=v2-alpha-tpuv6e        # JAX/Pallas 표준
PROJECT=tpu-board

gcloud compute tpus tpu-vm create $TPU_NAME \
  --project=$PROJECT \
  --zone=$ZONE \
  --accelerator-type=$ACCEL \
  --version=$RUNTIME \
  --preemptible                # 7일 자동 종료, ~1/4 가격
```

**소요 시간**: 보통 7-10분 (큰 슬라이스일수록 더 걸림). europe-west4-a 첫 생성 시 ~7분 관측.

**예상 비용** (v6e-1):
- Preemptible: ~$0.15/hr ≈ $3.60/일
- On-demand: ~$0.34/hr ≈ $8/일
- 더 큰 슬라이스 가격: chip 수에 거의 선형 비례

### 3.2 에러 핸들링

**`Failed to perform tenant project creation`**:
- 원인: (a) service identity 미생성, (b) zone-level capacity 부족
- 해결: 1.3의 service identity 명령 실행 + 다른 zone 시도 (us-east5-a, europe-west4-a 추천)
- asia-northeast1-b는 일시적으로 capacity 없을 수 있음. EU/US로 fallback

**상태 확인**:
```bash
gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT
gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE --project=$PROJECT \
  --format="value(state,acceleratorType,runtimeVersion,networkEndpoints[0].accessConfig.externalIp)"
```
`READY` 나오면 SSH 가능.

---

## 4. SSH 접속

### 4.1 gcloud 통한 접속 (가장 쉬움)
```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT
```
첫 접속 시 SSH 키 propagation 1-2분.

### 4.2 외부 SSH (VS Code Remote, scp 등)
```bash
# 외부 IP 확인
EXT_IP=$(gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE --project=$PROJECT \
  --format="value(networkEndpoints[0].accessConfig.externalIp)")
echo $EXT_IP

# 본인 공개키 등록 (한 번만)
gcloud compute os-login ssh-keys add --key-file=~/.ssh/id_rsa.pub

# ~/.ssh/config에 항목 추가
cat >> ~/.ssh/config <<EOF
Host tpu-v6e
  HostName $EXT_IP
  User okkyun_w_gmail_com
  IdentityFile ~/.ssh/id_rsa
EOF

ssh tpu-v6e
```

---

## 5. JAX 환경 셋업 (TPU VM 안에서)

기본 이미지 `v2-alpha-tpuv6e`에는 JAX 미설치 상태:

```bash
# TPU VM 안에서
pip install -U "jax[tpu]"
pip install tensorboard tensorboard-plugin-profile

# 동작 확인
python3 -c "
import jax
print('JAX:', jax.__version__)
print('Devices:', jax.devices())
print('Count:', jax.device_count())
"
# 기대 결과: TpuDevice(id=0, ...) 표시
```

---

## 6. 워크로드 + Profile 캡처

### 6.1 표준 워크로드 스크립트 템플릿

다음 패턴을 따르면 됨:
```python
import jax, jax.numpy as jnp

# (모델 정의 + JIT)
jit_forward = jax.jit(forward)

# 워밍업
jit_forward(params, tokens).block_until_ready()

# 프로파일 캡처
import os
logdir = "/tmp/tb_logs"
os.makedirs(logdir, exist_ok=True)
with jax.profiler.trace(logdir):
    for _ in range(5):
        out = jit_forward(params, tokens)
    out.block_until_ready()
```

생성 파일:
- `/tmp/tb_logs/plugins/profile/<timestamp>/*.trace.json.gz` (Chrome trace)
- `/tmp/tb_logs/plugins/profile/<timestamp>/*.xplane.pb` (XLA structured profile)

### 6.2 (선택) 컴포넌트별 라벨링 향상

코드에 `jax.named_scope` 추가하면 trace에 prefix가 붙어 분석이 깔끔해짐:
```python
def attn(x, layer):
    with jax.named_scope("attn_qkv"):
        q = (x @ layer['wq']).reshape(...)
        ...
    with jax.named_scope("attn_scores"):
        scores = q @ k.transpose(...) / jnp.sqrt(DH)
    with jax.named_scope("attn_softmax"):
        p = jax.nn.softmax(scores, axis=-1)
    with jax.named_scope("attn_pv"):
        out = (p @ v).transpose(...).reshape(...)
    with jax.named_scope("attn_out_proj"):
        return out @ layer['wo']
```

또는 Flax/NNX/Penzai 같은 module 기반 코드는 자동으로 module path가 op name에 들어감.

---

## 7. Profile 데이터 로컬로 다운로드

```bash
# 로컬에서 실행
mkdir -p /tmp/tb_logs_local
gcloud compute tpus tpu-vm scp --recurse \
  $TPU_NAME:/tmp/tb_logs/ \
  /tmp/tb_logs_local/ \
  --zone=$ZONE --project=$PROJECT
```

검증:
```bash
find /tmp/tb_logs_local -type f
# 기대: *.trace.json.gz, *.xplane.pb
```

---

## 8. 로컬 TensorBoard 실행

```bash
# 의존성
pip3 install --user tensorboard tensorboard-plugin-profile

# 띄우기 (백그라운드)
nohup python3 -m tensorboard.main \
  --logdir=/tmp/tb_logs_local --port=6006 \
  > /tmp/tb_local.log 2>&1 &

# 동작 확인
sleep 6
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:6006/
# 기대: HTTP 200
```

브라우저: <http://localhost:6006> → 상단 **PROFILE** 탭

주요 도구 (좌측 "Tool" 드롭다운):
- `overview_page` — 한눈에 보기 (compute/memory/idle %, top ops)
- `op_profile` — 연산자별 시간/utilization
- `trace_viewer` — Chrome trace 타임라인
- `memory_viewer` — HBM 사용량
- `graph_viewer` — XLA HLO 그래프

종료: `pkill -f "tensorboard.main"`

---

## 9. 분석 — 핵심: trace.json.gz 직접 파싱

per-op latency는 **trace.json.gz**에서 직접 추출 가능 (TensorBoard 안 거치고).

### 9.1 구조

```python
import gzip, json

with gzip.open("PATH/t1v-n-XXXX.trace.json.gz", "rt") as f:
    data = json.load(f)

# pid=3, tid=3 = "/device:TPU:0" / "XLA Ops" (실제 디바이스 op 단위)
ops = [e for e in data["traceEvents"]
       if e.get("ph") == "X" and e.get("pid") == 3 and e.get("tid") == 3]
```

각 op 이벤트의 유용한 필드:
- `name`: HLO op 이름 (예: `fusion.89`)
- `dur`: 실행 시간 (μs)
- `args.model_flops`: FLOPs
- `args.bytes_accessed`: HBM 트래픽 (bytes)
- `args.shape_with_layout`: 출력 shape + layout
- `args.long_name`: 원본 HLO 명령어
- `args.hlo_category`: 분류 (convolution fusion / loop fusion / copy 등)
- `args.source`: **Python 소스 위치** (file:line) ← 컴포넌트별 분류에 결정적
- `args.tf_op`: TF op 이름 (있을 시)

### 9.2 컴포넌트별 latency 분리 (검증된 방법)

`args.source` 필드를 활용해 Python line별 그루핑:
```python
import collections
by_src = collections.defaultdict(lambda: {"dur":0, "n":0, "flops":0, "bytes":0})
for e in ops:
    src = e["args"].get("source", "")
    if not src: continue
    loc = src.split(",")[0]
    by_src[loc]["dur"] += e["dur"]
    by_src[loc]["n"] += 1
    by_src[loc]["flops"] += int(e["args"].get("model_flops", 0) or 0)
    by_src[loc]["bytes"] += int(e["args"].get("bytes_accessed", 0) or 0)
```

### 9.3 GEMM-size별 BW/효율 표

```python
# v6e 사양
HBM_PEAK_GBPS = 1638 * (2**30) / 1e9  # ≈ 1759 GB/s
PEAK_TFLOPS = 918                      # BF16
# Roofline crossover: ~522 FLOP/byte

for e in ops:
    flops = int(e["args"].get("model_flops", 0) or 0)
    if flops < 1e8: continue
    bytes_ = int(e["args"].get("bytes_accessed", 0) or 0)
    dur = e["dur"]
    tflops = flops / (dur*1e-6) / 1e12
    bw = bytes_ / (dur*1e-6) / 1e9
    print(f"{e['name']:30s}  {tflops:6.1f} TFLOPS ({100*tflops/PEAK_TFLOPS:5.1f}%)  {bw:7.1f} GB/s ({100*bw/HBM_PEAK_GBPS:5.1f}%)")
```

### 9.4 검증된 분석 결과 (한 번 돌렸을 때)

이전 세션에서 `B=16, S=512, D=1024, L=8, H=8, DH=128, V=32000` 모델 1 forward pass:
- 총 wall time: **4.57 ms / iter** (5 iter 평균)
- 분포 (per iter):

| 컴포넌트 | μs | % | TFLOPS | 비고 |
|---|---|---|---|---|
| FFN (w1+GELU+w2 fused) | 1555 | 34% | 709 | compute-bound |
| LM_HEAD | 686 | 15% | 783 | 85% peak ⭐ |
| Attention PV | 551 | 12% | 128 | memory-bound (93% HBM) |
| Attention QKᵀ | 383 | 8% | 182 | memory-bound |
| Softmax | 382 | 8% | 3.5 | pure memory-bound |
| V proj | 243 | 5% | 566 | |
| Output proj | 239 | 5% | 577 | |
| Q proj | 215 | 5% | 641 | |
| K proj | 198 | 4% | 695 | |
| Embed lookup | 115 | 3% | 0 | mem-only |
| RMSNorm | 4.5 | 0.1% | 6 | |

핵심 관찰:
- **Attention 컴포넌트 (QKᵀ + Softmax + PV) = 1316 μs/iter = 29%**, 대부분 memory-bound → Flash attention 도입시 큰 이득
- **Q vs K vs V projection**이 같은 shape인데 ±20% 차이 — 후속 reshape/transpose가 attribute된 결과
- HBM 사용률 attention 계열에서 87-93% 도달 (XLA가 메모리 한계까지 뽑음)

---

## 10. **삭제** (CRITICAL — 절대 잊지 말 것)

작업 끝나면 **즉시 삭제**. 안 그러면 매시간 과금:

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME \
  --zone=$ZONE --project=$PROJECT --quiet
```

확인:
```bash
gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT
# 빈 결과면 성공
```

비용 사후 확인:
- Billing 콘솔: <https://console.cloud.google.com/billing/010005-50DF18-C442F4/reports?project=tpu-board>

---

## 11. 알려진 함정

1. **JAX 미설치**: `v2-alpha-tpuv6e` 이미지에 JAX 없음. `pip install -U "jax[tpu]"` 필요
2. **Zone capacity 변동**: 검증된 zone도 일시적으로 못 만들 수 있음. fallback zone 리스트 준비:
   - 1순위: `europe-west4-a` (검증됨)
   - 2순위: `us-east5-a`, `us-south1-a`
   - 3순위: `asia-northeast1-b` (한국 가까움, 종종 capacity 없음)
3. **Preemptible 종료**: 7일 후 자동 종료, 그 안에서도 회수 가능. 진지한 학습/실험엔 on-demand
4. **TensorBoard 명령어**: `tensorboard` 바이너리가 PATH에 없을 수 있음 → `python3 -m tensorboard.main` 사용
5. **외부 SSH OS Login 이메일**: `okkyun.w@gmail.com` → `okkyun_w_gmail_com` (점/@ 변환)
6. **profile 캡처 시 너무 짧으면 빈 trace**: `block_until_ready()` 호출 + 최소 5 iter 권장
7. **`jax.profiler.trace()`는 context manager**, `start_trace()/stop_trace()` 페어도 있음

---

## 12. 다음 단계 권장 워크로드

이번에 못 본 것:
- **Backward pass + optimizer step** (training): forward의 ~2-3배 비용. 시뮬에 필수
- **다양한 (B, S, D) sweep**: efficiency 곡선 fitting (각 ~5분, 무인 자동화 가능)
- **실제 모델** (HF GPT-2, Llama 등): realistic op mix
- **Multi-chip (v6e-8+)**: ICI 통신 cost (단일 칩엔 없음)
- **KV-cache decode**: memory-bound 행동 측정

---

## 부록 A. 정보 한 줄 요약 (복붙용)

```
PROJECT=tpu-board
BILLING=010005-50DF18-C442F4
ACCOUNT=okkyun.w@gmail.com
ZONE=europe-west4-a
RUNTIME=v2-alpha-tpuv6e
PEAK_HBM_GBPS=1759
PEAK_BF16_TFLOPS=918
ROOFLINE_AI=522  # FLOP/byte
```

## 부록 B. v6e 공식 사양 (참고)

| 항목 | 값 |
|---|---|
| Peak BF16 TFLOPS | 918 |
| Peak INT8 TOPS | 1,836 |
| HBM capacity | 32 GB / chip |
| HBM bandwidth | 1,638 GiB/s ≈ 1,759 GB/s / chip |
| ICI bidirectional | 800 GB/s / chip (4 ports) |
| Chips per host | 8 |
| Pod max | 256 chips |
| MXU | 256×256 (이전 세대 128×128에서 증가) |
| Clock | **미공개** |
| VMEM size | **미공개** |
| CMEM | **명시적 비공개** |
| VREG | **미공개** |

(시뮬레이션 calibration시 미공개 항목은 v5p/v5e 값 차용 또는 Pallas로 실측)
