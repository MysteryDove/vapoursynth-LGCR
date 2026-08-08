# LGCR — Luma-Guided Chroma Reconstruction

一个研究型 VapourSynth 4 插件：用**亮度引导的方向自适应卷积核**重建/缩放色度，
解决 YUV 子像素彩边（紫边/绿边/色度锯齿）问题。CPU 实现，AVX2 FMA 快速路径，
全部引导项为有理函数（无 exp/div），对 AVX-512 友好。

当前形态（v1.10）：

```
W'uv = Sim(ΔL) · ( W_kernel + rescue(φ) · Bump_aniso(θ) )       # algo 2/4
C1   = C0 + g · a · R(lp_axis(Y − P(median_k D_k(Y))))          # algo 6（细节迁移）
```

- 基核可选 bilinear / bicubic / lanczos / spline16 / spline36 / **jinc**（全 2D 径向路径）
- 边缘方向 θ 与各向异性来自**亮度结构张量**（非裸 Sobel）
- 3 种重建算法入口（`algo=2/4/6`）+ 独立的边缘感知锐化函数 `lgcr.Sharpen`
  + 时域重建函数 `lgcr.TRecon`（运动补偿多帧相位分集）
- v1.6 语义修正：连续 strength、逻辑 tap 掩码、signed wsum 防护、按核族的
  rescue 门控、H.273 `_ChromaLocation` 支持、稀疏引导、可选 back-projection
- v1.8（外部评审驱动）：**mutual-structure 共边门控**（`ms`，修复 hardL_softC
  残差与 misalign4 语义）；TRecon 阻断级修正（整数归一化反转、参数崩溃、
  相位奇偶反转、边界帧伪收益、平坦块 ME 歧义）；plain 与引导统一代码路径
  （strength 精确连续）；algo3/4 缩放置信修复（upscale algo3 0.208→0.023）
- v1.9：**algo=6 受约束细节迁移**（源网格轴 Nyquist 抑制 + 法向 1D +
  `q_prod×ms` 复合置信）；门控统计量分类器评测基建（独立获益标签、三种实际
  退化核、ROC/风险覆盖）；时域指标改为对齐边缘带误差方差；新残差案例
  misalign1（亚色度相位错位）
- v1.10：收敛公开算法面为 `2/4/6`；移除已被替代的 algo1、风险过高的独立
  LGF algo3 和全线落败的 NEDI-lite algo5，algo4 继续内部使用 LGF 对角分支

---

## 目录

1. [安装与使用](#1-安装与使用)
2. [参数参考](#2-参数参考)
3. [算法](#3-算法)
4. [评测结果](#4-评测结果)
5. [关键设计决策与失败记录](#5-关键设计决策与失败记录)
6. [文献调研与空白点](#6-文献调研与空白点)
7. [性能与 SIMD](#7-性能与-simd)
8. [后续研究方向](#8-后续研究方向)
9. [文件结构](#9-文件结构)

---

## 1. 安装与使用

### 构建

```bash
cd bsflab
make            # 生成 liblgcr.so（AVX2）
                # VS 头文件路径用 VSINCLUDE 覆盖，默认读 ~/vapoursynth 下的安装
make liblgcr_scalar.so   # 可选：标量参考构建，用于正确性交叉验证
```

要求：g++ ≥ 11（C++17 `std::cyl_bessel_j`）、VapourSynth R55+（API v4）头文件、
CPU 支持 AVX2+FMA（标量构建无此要求）。

### 加载

```python
import vapoursynth as vs
core = vs.core
core.std.LoadPlugin("/home/owen/dev/bsflab/liblgcr.so")
```

或者把 `liblgcr.so` 放进 VS 的自动加载目录（Linux 通常是
`~/.local/lib/vapoursynth/` 或 `/usr/local/lib/vapoursynth/`），之后无需
`LoadPlugin` 即可直接用 `core.lgcr.*`。

### 两个函数

插件提供 `Recon`（引导色度重建/缩放）和 `Sharpen`（边缘感知锐化）。

### 典型用法

**A. 420 → 444 色度重建（最常见场景：解码后送渲染器前）**

```python
src = core.ffms2.Source("input.mkv")          # YUV420P8/10 解码
out = core.lgcr.Recon(src)                    # 输出同尺寸 YUV444，默认 algo=2 + lanczos3
```

**B. 缩放 + 引导色度重建（如 1080p → 4K）**

```python
out = core.lgcr.Recon(src, width=3840, height=2160, kernel="jinc", taps=3)
# Y 平面：同一核做高质量重采样；U/V：引导重建到输出网格
```

**C. A/B 对照（验证引导在你素材上的效果）**

```python
guided = core.lgcr.Recon(src, kernel="jinc", taps=3)
plain  = core.lgcr.Recon(src, kernel="jinc", taps=3, strength=0)  # 纯核，引导全关
diff   = core.std.MakeDiff(guided, plain)
```

**D. 算法入口选择**

```python
out = core.lgcr.Recon(src, algo=2)   # 默认，真实视频最稳
out = core.lgcr.Recon(src, algo=4)   # 逐像素选择器，硬色边素材（屏幕录制/图形/字幕）更佳
out = core.lgcr.Recon(src, algo=6)   # 受约束细节迁移（实验性，见 §3.3）
```

**E. 时域重建（动画/多帧素材）**

```python
out = core.lgcr.TRecon(src420, trad=1)            # 前后各 1 帧
out = core.lgcr.TRecon(src420, trad=2, tsearch=8) # 更大窗口/搜索范围
```

机制：亮度块匹配（16×16，整数像素，SAD）把邻帧色度样本对齐到当前帧坐标，
并以块内运动补偿联合色度差的中位数拒绝换色邻帧，再把通过的样本作为额外 tap
注入引导权重循环。**相位分集**：**奇数**亮度像素位移使 420 采样
相位平移半个色度像素，邻帧样本携带真正的新采样信息；偶数位移（整数色度
像素）只是同相位重采样，对运动块被精确门控为零（t_move2 与单帧 Recon 完全
持平——不再虚报收益）。静态块（|mv|≤1 且高置信）的冗余 tap 是纯时域平均
降噪（t_static_noise 0.0263 < 单帧 0.0285，全场最佳）。平坦亮度块的运动不可
观测（孔径问题），其置信度自动归零，不会拖动等亮度色度纹理。边界帧不做
重复利用：单帧 clip 的 TRecon 输出与 Recon 逐位一致（隔离测试 maxdiff=0）。
仅同尺寸 420→444；静态场景多的动画收益最大。

**F. 边缘感知锐化（gated Laplacian，硬边零 halo）**

```python
out = core.lgcr.Sharpen(clip, alpha=0.4)   # 任意 YUV 格式，同格式输出
# 推荐链路：Recon 之后、编码之前
final = core.lgcr.Sharpen(core.lgcr.Recon(src, algo=4), alpha=0.3)
```

**G. 与 zimg（内置 resize）分工**：`Recon` 的输出是 444；如需回 420 交付，
用 `core.resize.Spline36(out, format=vs.YUV420P10)` 做最终下采样即可。

### 用 vspipe 跑批

```bash
vspipe --y4m script.vpy - | ffmpeg -i - -c:v libx265 -crf 16 out.mkv
```

### 静态图测试管线

```bash
~/vapoursynth/bin/python3 test/recon_image.py in.png out.png jinc 0.8
# PNG -> 模拟 420 退化 -> LGCR 重建 -> PNG，可指定 kernel/strength/scale
```

---

## 2. 参数参考

### `lgcr.Recon`

输入：planar YUV 8–16 bit int 或 32 bit float，4:4:4 / 4:2:2 / 4:2:0。
输出：同位深同家族的 **YUV 4:4:4**。

| 参数 | 默认 | 含义 |
|---|---|---|
| `width`/`height` | 输入尺寸 | 输出尺寸（不填 = 纯色度重建模式） |
| `loc` | 未设时读 `_ChromaLocation` | 水平 siting；垂直 siting 由 H.273 属性（topleft/top/bottomleft/bottom）自动处理；输出 444 会删除该属性 |
| `kernel` | `"lanczos"` | `"bilinear"` / `"bicubic"`(b,c 可调) / `"lanczos"`(taps) / `"spline16"` / `"spline36"` / `"jinc"`(taps) |
| `taps` | 3 | lanczos/jinc 的瓣数，接受范围 `1..64`；越界作为插件参数错误返回 |
| `algo` | 2 | 重建算法：`2`=引导选择（默认）；`4`=逐像素混合选择器；`6`=受约束细节迁移（实验性）。其他编号返回参数错误 |
| `strength` | 0.8 | 引导/修正强度 λ；**0 = 纯核**（A/B 对照基准） |
| `sigma` | 0.01 | 亮度相似度 σ 下限（归一化单位），相当于噪声地板 |
| `sratio` | 0.15 | 自适应拐点：`sig ≥ sratio · 窗内亮度range` |
| `sdb` | 3.0 | 斜坡时的 σ 倍数（×窗内range） |
| `stretch` | 1.0 | 沿边缘支撑拉伸倍数（EWA 各向异性，0 = 各向同性） |
| `gsigma` | 2.5 | 引导 bump 宽度（亮度像素），跨边缘方向 |
| `ridge` | 1 | 细线稿（亮度 ridge）检测开关：检出时回退纯核（algo 2/4） |
| `ms` | 1.0 | **mutual-structure 共边门控**强度 [0,1]（algo 2/4/6，连续淡化）：色度须在方向/位置/宽度上确认与亮度共边，否则禁止亮度高频转移。修复 hardL_softC（软色边不再被锐化）与 misalign4（错位色边不再被搬移） |
| `qgate` | 1.0 | algo=6 的仿射可信度门控强度 [0,1]；`0` 禁用 q 门但保留细节迁移，`1` 使用完整 `qMean·stability·sigC`，主要用于消融/评测 |
| `cedge` | 0 | **实验性**：色度过渡宽度检测。实测估计量不稳定（详见 §5），默认关闭；其目标场景已由 `ms` 解决 |
| `ar` | 0.0 | anti-ringing：输出 clamp 到窗内色度 hull + 此余量；<0 关闭 |
| `reg` | 0.005 | algo4 内部 LGF 分支及 algo6 回归的正则化，ε = reg² |
| `sparse` | 1 | 稀疏引导：纯核全帧 + 仅亮度结构区域跑引导（约 10× 加速，输出与密集模式逐位一致的区域占绝大多数） |
| `bp` | 0.0 | back-projection 一致性增益（仅同尺寸 420）：`D_h(C)≈C_src` 的一阶修正。**注意**：D_h 目前是 2×2 box，与 bilinear 类降采样器不匹配时反而有害——仅在确认编码器用 box 下采样（如 JPEG 系）时使用 |

### `lgcr.Sharpen`

输入/输出同格式（任意 planar YUV，同尺寸）。

| 参数 | 默认 | 含义 |
|---|---|---|
| `alpha` | 0.3 | 锐化强度（HP 项系数）；实测响应温和，0.5–1.0 也安全 |
| `sigma` | 0.01 | 相似度 σ 下限（噪声地板；抬到噪声幅度以上则平坦区零放大） |
| `sratio` | 0.15 | 自适应拐点比例（同 Recon） |
| `gspatial` | 1.2 | 空间 bump 宽度（平面像素），5×5 窗 |
| `ar` | 0.0 | hull clamp 余量；<0 关闭 |

机制：`HP = Σ n_i(x₀−x_i)/Σ n_i`，`n_i = 空间bump × sim(ΔL)`；`out = x₀ + α·HP`。
硬边处跨侧 tap 被 sim 清零 → HP≈0 → **结构性无 halo**；软过渡带肩部有同侧
曲率 → 过渡带变陡。Y 自引导，色度由亮度引导。

### `lgcr.TRecon`

输入：planar YUV（同 Recon），恒定尺寸与帧数。输出：同尺寸 YUV 4:4:4。

| 参数 | 默认 | 含义 |
|---|---|---|
| `trad` | 1 | 时域半径（每侧帧数，总参考 2·trad+1 帧） |
| `tsearch` | 6 | 块匹配搜索范围（亮度像素，整数） |
| `tsad` | 0.02 | ME 匹配置信度尺度（SAD/像素，归一化） |
| `strength`/`sigma`/`sratio`/`sdb`/`gsigma`/`stretch`/`ar`/`ridge`/`ms` | 同 Recon | 引导参数，语义同 Recon（algo=2 路径） |
| `sparse` | 0 | TRecon 默认密集（静态区的时域平均降噪是收益的一部分） |

时域 tap 权重：`w = tconf · cconf · sim · W_kernel(真实偏移) · 相位新颖度`。
- `tconf` = 匹配质量 × 可观测性（亮度块方差；平坦块运动不可观测 → 0）；
- `cconf = 1/(1+(medianDelta/0.04)^4)`，`medianDelta` 是每个 16×16 亮度块内
  运动补偿后 `max(|ΔU|,|ΔV|)` 的中位数；`tconf·cconf < 0.05` 时整块邻帧 tap 被拒绝；
- 运动块的 `sim` 用**源亮度 footprint 骑跨测试**（运动补偿后 footprint 内任一
  亮度样本与 L0 的最大差）× 0.5σ 裸 sim——色度分辨率亮度水平对边缘列的
  部分骑跨是盲的，这是奇数位移曾回归 0.0234 的根因；
- `W_kernel(真实偏移)`：时域样本就是非格点位置的空间样本，按基核取值，
  不再是恒定 ~0.6 的胖 bump；
- 静态块（|mv|≤1 且 tconf>0.5）新颖度=1（纯平均）；运动块仅当位移在色度
  单位下的小数部分 ≈½（奇数亮度像素）时新颖度=1，整数色度像素位移=0。

---

## 3. 算法

### 3.1 主导引路径（algo=2，默认）

对每个输出色度样本（输出 4:4:4 网格 = 输出亮度网格），设重建中心在源色度坐标
`(scx, scy)`，支持窗内第 i 个 tap 的偏移为 `d_i`（换算为亮度像素单位）：

```
L0      = 双线性采样 *全分辨率* Y，位置 = 重建中心的精确亮度坐标
J       = 亮度结构张量（Sobel 外积的 3×3 盒平滑）
n̂       = J 主特征向量（边缘法向）；  coherence = (λ₁−λ₂)/(λ₁+λ₂)
minDiff = min(|Y(L0+n̂·sp)−L0|, |Y(L0−n̂·sp)−L0|)      # 法向最小邻差（sp=tap间距）
sig     = max(sigma, max|dL| · lerp(sratio, sdb, ss))  # ss=smoothstep(minDiff/0.1R)
sim_i   = 1 / (1 + ((Lc_i − L0)/sig)²)                 # 亮度相似度（Lorentzian）
a       = 1 + stretch · coherence                      # EWA 各向异性
bump_i  = 1 / (1 + (dperp/gsigma)² + (dpar/(gsigma·a))²)
rescue  = strength · 相位接近度² · (1 − sim_重合tap)     # 相位0救援项
fade    = ridge淡化 × ms共边门控（细线稿/软色边/错位边回退纯核）
sim_i  ← 1 − fade·(1 − sim_i)；  rescue ← fade · rescue
W'_i    = sim_i · (W_kernel,i + rescue·bump_i)
C_out   = Σ W'_i·C_i / Σ W'_i                          # 归一化保直流
C_out  ← clamp(C_out, hull_min − ar, hull_max + ar)    # anti-ringing
```

其中 `Lc_i` 是第 i 个色度采样点在其 **sited footprint** 上的亮度均值（420 left
siting 时 footprint 为对应 2×2 亮度块，支持 `loc=left/center`）。U/V 共享全部
引导权重，单次计算双平面累加。

fade 中的 ms 共边门控（mutual-structure，色度分辨率，逐色度样本预计算）：

```
沿亮度边缘法向取 ±3 色度像素的梯度幅度轮廓 gY(k)、gC(k)
partY = (ΣgY)²/ΣgY²，partC = (ΣgC)²/ΣgC²      # 参与率：边缘展宽的 tap 数
dcent = |centroid(gY) − centroid(gC)|           # 轮廓质心相位差（色度 px）
gate  = smoothstep((1.6 − partC/partY)/0.4) · smoothstep((1.0 − dcent)/0.5)
```

直觉：亮度硬边下的**软**色度混合（hardL_softC）partC/partY≈1.8 → gate 0；
错位 4 亮度像素的色边（misalign4）dcent≈2 → gate 0；严格共边的硬边
ratio=1.00、dcent=0 → gate 1。亮度侧用 footprint 盒平均的 lc 图（即候选 420
编码器滤波器），比较发生在色度信息真实存在的分辨率上。

### 3.2 各 algo 一览

| algo | 机制 | 一句话 |
|---|---|---|
| 2 | v1.3+ST：sim 选择 + 法向探针判别 + ridge/ms 淡化 | **默认**，真实内容最稳 |
| 4 | 逐像素选择器：硬边程度 × 轴/对角 → 路由 algo2/内部 LGF/纯核 | 图形型硬边档；对角硬边明显强于 algo2，其余场景贴 plain |
| 6 | 受约束细节迁移：`C1 = C0 + g·a·detail_safe`（下述） | 合成电池强（d45 0.0605、upscale 0.0251、noise 0.0295）；单张真实动画也优于 plain，仍属实验性 |

历史 `algo=1/3/5` 已从代码和公开接口移除。编号 `2/4/6` 保持不变，避免现有脚本
在升级后静默切换到不同机制。

### 3.3 algo=6：受约束细节迁移（针对性别名抑制）

不替换纯核输出，只补它丢掉的细节：

```
C0 = P(C420)                              # 纯核基底（低频与颜色基准）
Yc = median_k D_k(Y)，Yb = P_source(Yc)   # 多核共识亮度，在源亮度网格重建
detail_safe = R(lp_axis(Y − Yb))          # 先抑制轴别名，再缩放到输出网格
C1 = C0 + g · a · detail_safe
```

- `a`：色度网格 5×5 窗回归斜率 cov(Yc,C)/var(Yc)，对候选退化核
  {box, triangle, bicubic} 分别估计后取中值；斜率和共识亮度不由 q 选择，因而
  `qgate=0` 能给出独立于待评估门控的校正标签。
- `g`：生产置信为 `qMean · (qmin/qmax) · sigC`，其中每个 q 是 U/V 联合仿射
  可信度 `(covU²+covV²)/((varY+ε)(varU+varV+ε))`；应用时再乘 **ms 共边门**
  与 strength。`qgate`/`ms` 都按 [0,1] 连续插值，不是布尔开关。
- `lp_axis` + `R`：先在**源亮度网格**对每个被子采样轴施加 [1,2,1]/4，精确
  消除该轴的 Nyquist 交替模，再缩放到输出；随后按结构张量 coherence 做切向
  平滑，只保留更接近法向 1D 的细节。横、纵、棋盘三个定向反例在原尺寸和 2×
  输出均与无纹理基线相差 <0.0001。
- 幅度上限（|ΔC| ≤ ½·窗内色度 range）+ hull clamp。strength=0 ≡ 纯核逐位一致。

这里的 [1,2,1] 是**针对已知轴 Nyquist 风险的抑制器**，不是未知编码器完整
零空间的正交投影。斜向/非周期不可观测模式仍需更广的反例族验证，文档和代码
不再使用“零空间安全”这一过强表述。

与 algo4 的内部 LGF 候选不同，algo6 不用 `a·Y+b` 整体替换色度：其低频/颜色
来自纯核，仿射模型只提供缺失的高频细节，且细节先经有针对性的不可观测频率抑制。

### 3.4 Sharpen（gated Laplacian）

`HP = Σ n_i(x₀−x_i)/Σ n_i`（`n_i = 空间bump × sim`），`out = x₀ + α·HP`，
hull clamp。门控拉普拉斯：边缘处 HP≈0（无 halo 的来源），软过渡带肩部收紧。
σ 含一阶斜率地板（`max(sigma, sratio·range, 3·slope_med)`）——斜坡不被误当作边缘。

---

## 4. 评测结果

### 可复现评测电池（test/battery.py，v1.9）

`python3 test/battery.py [--all] [--check] [--write-results]`。`--check` 启用质量
断言且不写文件；只有显式 `--write-results` 才更新 `test/results/latest.md`。
当前成绩（jinc3，strength=0.8；strength 是连续的，0.8 ≠ 全强度）：

| case | plain | algo2 | algo4 | algo6 |
|---|---|---|---|---|
| hard_v（严格共边硬边） | 0.0363 | 0.0236 | **0.0218** | 0.0245 |
| hard_d45（对角硬边） | 0.0881 | 0.0797 | **0.0413** | 0.0605 |
| isoluminant（无亮度边色边） | 0.0363 | 0.0363 | 0.0363 | 0.0363 |
| misalign4（Y/C 错位 4px） | 0.0203 | 0.0203 | 0.0203 | 0.0203 |
| misalign1（亚色度相位错位 0.5cp） | **0.0156** | 0.0334 | 0.0360 | 0.0173 |
| nullspace_x（横向 Nyquist） | 0.0363 | 0.0363 | 0.0363 | **0.0245** |
| nullspace_y（纵向 Nyquist） | 0.0363 | 0.0596 | 0.0568 | **0.0246** |
| nullspace_xy（棋盘 Nyquist） | 0.0363 | 0.0363 | 0.0363 | **0.0245** |
| hardL_softC（硬亮度边+软色度） | 0.0056 | 0.0059 | 0.0057 | **0.0056** |
| ridge_line（线稿） | **0.0018** | 0.0018 | 0.0018 | 0.0019 |
| ramp | 0.0051 | 0.0051 | 0.0055 | 0.0051 |
| texture | 0.0120 | 0.0120 | 0.0120 | 0.0120 |
| noise（σ=0.008） | 0.0375 | 0.0376 | 0.0381 | **0.0295** |
| upscale2x | 0.0386 | 0.0337 | 0.0329 | **0.0251** |
| temporal（对齐边缘带误差方差） | 0.00008 | 0.00007 | 0.00005 | **0.00004** |

misalign1 仍揭示亚色度像素 Y/C 相位错位会把边缘搬错位置：algo2 明显退化，
algo6 经源网格约束后只剩 `+0.00173`，但仍未胜过 plain。答案仍是路线图上的
相位补偿，而不是继续收紧门控。三个 nullspace 用例是定向 Nyquist 反例，不代表
完整零空间；已移除的独立 LGF 基线会灾难性注入，algo6 的轴向抑制使三者都
回到正常硬边收益。

### 门控统计量的分类器评测（test/eval_gates.py）

运行 `python3 test/eval_gates.py`，完整输出写入
`test/results/eval_gates_latest.md`。

获益标签 = "启用机制相对纯核是否获益"（非人为硬/软语义）。当前是 156 个互异
样本（软度 × 亚色度相位错位 × 0°/22.5°/45° × 对比度 + 反例族），每个分别用
box / triangle / bicubic 生成实际 420，共 468 个观测。q 使用 C++ 同定义的总体
协方差、`reg²=2.5e-5`；风险覆盖同时报告阈值并列造成的实际覆盖率。

对"无门控吸附"标签（algo2 `ms=0` vs plain，135 获益 / 285 伤害）：旧 cedge
估计器的色度网格 proxy AUC `0.632`，旋转等变 + 单调性约束的**当前校正宽度
基线** `0.664`，ms `0.711`。宽度基线随真实 D 的 AUC 从 box `0.445` 到 bicubic
`0.833`，因此这些数值既不是生产 cedge 的逐像素复现，也不能称理论天花板。

对"细节迁移"标签，先用 `algo=6, qgate=0, ms=0` 生成与待测门控独立的标签
（205 获益 / 236 伤害）：`q_box=0.745`、pixelwise `q_min=0.732`、生产
`q_prod=0.724`、`ms=0.769`、逐像素 `q_prod×ms=0.777`。复合分数在前
5%/20%/50% 覆盖的平均风险分别为 `-0.0066/-0.0054/-0.0025`；完整 algo6
相对 plain 的全体平均 delta 为 `-0.00122`，说明它适合风险排序，但远非可靠的
二分类器。旧 `0.925` 来自 q 未真正关闭、乘的是带均值而非逐像素生产门控、且
只用一种实际退化核，现已作废。

按实际 D 分组时，`q_prod×ms` AUC 为 box `0.982`、triangle `0.990`、bicubic
`0.660`。多核 `min` 在总体上没有胜过单 box；当前瓶颈是跨退化核标定漂移，
下一步应先推断/显式指定编码器 D，再做条件门控，而不是继续调一个全局阈值。

时域用例（plain / algo2 / **TRecon**）：

| case | plain | algo2 | TRecon | 解读 |
|---|---|---|---|---|
| t_move1 / t_move3（奇数位移，真实新相位） | 0.0248 | 0.0159 | 0.0160 | 与单帧近似持平（超锐边缘上部分骑跨 tap 被保守抑制） |
| t_move2（偶数位移，同相位） | 0.0248 | 0.0159 | 0.0159 | **精确持平**——偶数位移无新信息，不再虚报 |
| t_static_noise（静态+噪声） | 0.0278 | 0.0285 | **0.0263** | 时域平均降噪，全场最佳 |
| t_single（单帧隔离） | — | — | maxdiff **0.000000** | 边界帧钳制的重复 tap 不再计入 |
| t_flat_amb（相同帧+等亮度纹理） | — | 0.0122 | 0.0122 | 平坦块运动不可观测 → 零信任，不拖纹理 |

**诚实声明**：v1.7 曾报告 t_move2 TRecon 0.0061（2.6× 于单帧），外部评审
证明该收益来自边界帧重复 tap 与偶数位移动红包——非时域信息。修正后
TRecon 的可验证收益是静态降噪与"无退化保证"；奇数相位分集在超锐合成边缘
上与单帧持平（footprint 骑跨测试为安全牺牲了这部分理论收益）。

**strength 连续性验证**：回归集覆盖 `0、1e-6、0.000999、0.001、0.001001`；
跨旧 `1e-3` 边界的步长不超过总 correction 的 1% 加 `1e-7`。hull 始终包含
plain 基值，因此任意正 strength 只限制新增 correction，不会突然改写 plain 输出。

### 真实图像（动画截图 1920×1080，当前实现）

运行 `python3 test/eval_image.py /path/to/source.png`，结果写入
`test/results/real_image_latest.md`。8-bit BT.601 自洽 RGB↔YUV、bilinear 420 退化、
jinc3 重建；沿用历史口径：`|ΔxY|+|ΔyY|>30` 为边缘带（0.780%），`<3` 为
平滑带（43.628%）。输入 SHA-256：
`a8f86bd2053bfb9b0dc14ba494250b7cfd893b402cbef31c9cf69f7e12577af6`。

| | plain | algo2 | algo4 | algo6 |
|---|---|---|---|---|
| 全帧 | 0.383 | 0.385 | 0.390 | **0.380** |
| 边缘带 | 1.529 | 1.549 | 1.563 | **1.471** |
| 平滑带 | **0.325** | 0.326 | 0.331 | **0.325** |

解读：algo6 的 `0.380/1.471/0.325` 三项均不差于 plain；algo4 保留的是图形型
硬边专项价值，而不是这张真实图上的平均优势。单张图不足以证明真实分布泛化，尤其门控评测已暴露
bicubic 软边残差；在建立多作品、多编码器 D 的固定真实动画集之前，默认仍保持
algo2，algo6 保持实验性。

### f32 输入的影响

plain/algo2 微幅改善（0.349→0.333，少一次 8bit 舍入）。插件内部全程 f32 计算。

---

## 5. 关键设计决策与失败记录

这些是从实测踩坑中得出的，每条都比代码本身更有价值：

1. **保留带符号基核（`W·sim`），不要用全正 bump 替换它**。基核负瓣在平滑区
   承担"反卷积下采样模糊"的作用；全正 bump 做不到。v1.0 加性混合在合成硬边上
   MAE -50%，但真实图片平滑区输给纯核 2 倍以上。
2. **相位 0 救援项必须是加性的**。φ=0 时基核退化为 delta，乘性调制 `0×f=0`，
   被边缘污染的源样本会原样漏过（这恰是 420 重建最关键的样本）。
3. **中心亮度 L0 必须取自全分辨率 Y**。色度分辨率块平均在边缘处产生中间值，
   使 ΔL 判别完全反转（污染样本被保留、好样本被杀）。
4. **台阶/斜坡判别的唯一正确形式是法向最小邻差** `min(|Y(L0±n̂·sp)−L0|)`。
   中值 4 邻域差死在对角台阶（只有两个同侧邻居）；第二小值死在轴对齐斜坡
   （有两个零差）。**不要判别能被局部一阶结构解释的亮度差。**
5. **方向项只做沿边缘拉伸（EWA），不做对称惩罚**。`|d·n̂|` 形式的跨边惩罚
   会把同侧一个色度样本外的邻居也杀掉，而它们正是修复污染样本的支撑。
6. **不要叠加对比度相关的置信门控**。gN=g/(g+g0) 类门控在低对比度边缘
   （紫边恰恰常见于这种地方）会把判别整个关掉；自适应 σ 已隐式完成平区保护。
7. **cedge（色度边缘存在性检测）实测失败，默认关闭**。原理是区分"420 毁掉的
   硬色边"与"真实的软色度混合"；但过渡宽度估计（range/法向梯度）在离散网格上
   噪声太大——对角硬边法向梯度被系统性低估、中央差分在混合样本上低估梯度。
   修正旋转与单调性后，三种实际 D 汇总 AUC 也只有 0.664，且分组漂移很大。
   **该目标场景已由 `ms`（§3.1）用轮廓形状比较更稳地处理**；cedge 仅作历史
   基线。这里是当前合成集结果，不是信息论上限。
8. **LGF（回归路线）在真实图像上输给选择路线的根源是结构迁移**：`a·Y0` 项把
   亮度细节（量化台阶、压缩噪声）乘以斜率注入色度。LGCR"没把握就回退纯核"
   的保守性是真实内容上的关键优势。
9. **NEDI-lite 全线落败的原因**：单步实现把对角协方差系数用于轴排列四邻域
   （仅 1/4 相位几何正确）；4-tap 协方差模型阶数不足以表达纹理。正确实现需
   两阶段 NEDI + siting 校正采样格。
10. **Y 自引导（oracle）实验是阴性结果**：引导有用的前提是引导图携带比数据
    严格更多的带宽；Y 引导 Y 时没有额外信息，oracle 也无收益。这验证了 LGCR
    的适用前提（全分辨率 Y 引导半分辨率 UV 天然满足）。
11. **合成硬边测试必要但不充分**：v1.0 在合成测试上完美、在真实图像上翻车。
    任何改动必须同时过合成电池 + 真实内容回归。
12. **时域收益必须做隔离测试**。v1.7 的 t_move2"2.6× 提升"被外部评审证伪：
    单帧 clip（邻帧全部钳制为当前帧）得到同样的 0.0060——收益来自重复正权
    tap 改变空间核形状，不是时域信息。边界钳制帧现在直接跳过；`t_single`
    隔离用例（TRecon≡Recon，maxdiff=0）常驻电池。
13. **420 相位分集的奇偶方向**：**奇数**亮度像素位移才翻转采样相位（半色度
    像素），偶数位移是同相位重采样。新颖度必须 keyed on 位移在色度单位的
    小数部分；v1.7 错用最近取整残差，奖励的恰好是冗余的偶数情况。
14. **块匹配置信度 = 匹配质量 × 可观测性**。平坦亮度块的运动不可观测（孔径
    问题）：任何位移 SAD 都相等。best/second-best margin 不可用——直线边缘
    沿切向天然并列最优（全零 margin）会误杀真实边缘块；正确做法是以亮度块
    方差度量可观测性 + 零运动偏向 tie-break + 零运动吸附阈值。
15. **时域 tap 的 sim 要用源亮度 footprint 骑跨测试，不能用色度分辨率亮度
    水平**。在边缘半值列，两侧样本的 footprint 平均亮度都与 L0 相近（7% 部分
    骑跨只移动 lc 0.0125，σ=0.01 下不可分），level 测试失明；对运动补偿后
    footprint 内的源亮度逐点取 max|ΔL| 才能识别骑跨。配合 0.5σ 裸 sim
    （不经 guideFade 软化——时域 tap 是全有/全无的加性项）。
16. **共边确认要比较轮廓形状，不是相关或能量**。梯度幅度轮廓的归一化相关
    ρ 分不开硬/软色边（0.89 vs 1.00——位置上同峰就高分）；能量门控会把
     legitimate 弱边截顶（R=0.05 台阶 eY=0.8）。有效的是参与率宽度比
    （partC/partY：1.00 vs 1.80）+ 质心相位差（0 vs 2.0 chroma px）。
17. **plain 路径必须与引导路径共用同一套边界语义**。分离的快速重采样路径
    （截断核归一化 vs tap 钳制）使 strength 0 与 1e-6 在边界差 0.05。
    现在 plain ≡ reconstructChroma(strength=0)，连续性精确成立；代价是
    plain 慢约 2.5×（211 vs 79 ms/帧），留待逐输出像素 SIMD 优化。
18. **不可观测的亮度细节必须由信号约束处理，不能只靠低分辨率统计量**。
    轴 Nyquist 交替模对所建模的 2× 对称低通不可见，r²/q/多核一致性/BP 都
    无法证实它属于色度。algo6 先在源网格施加 [1,2,1]/4；横、纵、棋盘反例
    均由已移除独立 LGF 基线的 0.779–1.017 降到 algo6 的约 0.0245。它只证明三个定向模式
    被抑制，不证明未知 D 的完整零空间已投影掉。
19. **分类器评测必须把门控从标签生成器中真正关掉，并逐像素复现生产分数**。
    旧 `algo6(ms=0)` 仍含 q，导致低 q 样本先被门控成 neutral；旧 `mean(q)·mean(ms)`
    也不等于生产的 `mean(q·ms)`。改用 `qgate=0,ms=0` 独立标签并扫三种实际 D
    后，复合 AUC 从虚高的 0.925 回落到 0.777。
20. **亚色度相位错位（0.5 chroma px）仍是吸附机制的机制性失败**。misalign1
    上 algo2 相对 plain 恶化 `+0.01787`；新 algo6 只剩 `+0.00173`，但仍未解决
    物理相位。此类误差需要估计并平移迁移目标，不是继续收紧门控。
21. **未知 D 的问题首先是跨域标定，不只是多核取 min**。`q_prod×ms` 在实际
    box/triangle 内 AUC 接近 0.99，在 bicubic 只有 0.66；总体 `q_min` 还略逊于
    单 box。应先做片段级 D 推断或由用户/元数据显式指定，再做条件阈值与校准。

---

## 6. 文献调研与空白点

相关工作梳理（2026-08 检索）：

| 方法 | 代表工作 | 亮度信息的用法 |
|---|---|---|
| Joint Bilateral Upsampling | Kopf et al. 2007；Kaiming He 的 guided filter 论文中 JBF/GF 上采样章节 | 仅作 **similarity**（亮度差高斯权重） |
| Guided Filter | He, Sun, Tang 2010 | 局部线性模型，无方向性 |
| Edge-Directed Interpolation | Allebach & Wong (NEDI)；TI VPE 硬件 EDI | 边缘方向用于**二选一/软切换**，不进入卷积核权重 |
| Luma-referenced chroma upsampling | Fu & Jung (IEEE Access 2018, HDR/WCG) | 由全分辨率亮度相关性生成色度权重 |
| Luma-assisted chroma subsampling | Korhonen (ICME 2015, [代码](https://github.com/jarikorhonen/chroma_upsampling)) | 亮度-色度局部相关性 |
| Joint chroma down/upsampling | Wang et al. ([IEEE TCSVT 2016](https://dl.acm.org/doi/10.1109/TCSVT.2015.2461891)) | 亮度结构迁移到色度（index map） |
| Luminance-guided joint-bilateral EWA jinc | GALOSH-RAW ([arXiv 2607.03768](https://arxiv.org/html/2607.03768v1)) | **最接近本工作**：EWA jinc + 亮度 bilateral + hull anti-ringing |

**谨慎结论**：在上表覆盖的非系统检索中，尚未找到把高质量插值核
（Lanczos/Spline/Jinc）与解析的亮度边缘方向 θ、边缘强度、亚像素相位 φ 同时
反馈进色度核形状的同构方案；GALOSH 的 EWA jinc 最接近，但 luma 主要进入
bilateral 相似度项。这个结果只支持“可研究的组合与实现空档”，**不等于已经
证明学术空白或首创**；投稿前仍需系统综述、专利检索、与现代学习式 joint
upsampling/chroma restoration 方法对照，以及公开数据集上的消融。

LGCR 当前可检验的研究定位是 **JBU 的三个推广组合**——① 空间基核允许负瓣
（spline/jinc 而非纯高斯），所以能反卷积而不是只会模糊；② 空间核由引导图的
结构张量做 EWA 各向异性变形；③ 相位相关的 rescue 处理上采样网格特有的
"重合 tap 被污染"奇点。等价视角：每输出像素一个即时构造的、引导调制的、
归一化的 transposed convolution（与 dynamic filter network 同思想，但权重
生成器是解析公式而非 CNN）。

---

## 7. 性能与 SIMD

单线程性能（本机 AVX2，1080p → 1080p444，lanczos3，真实图像内容）：

| 路径 | 时间 |
|---|---|
| 纯核（strength=0） | ~211 ms/帧 |
| algo=2 引导 sparse=1（默认） | **~419 ms/帧** |
| algo=2 引导 sparse=0（密集） | ~644 ms/帧 |

注意 v1.8 起 plain 与引导共用同一重建路径（strength 连续性修复，§5-17），
纯核不再走独立快速重采样（旧 79 ms/帧）；性能追回列入路线图（逐输出像素
SIMD 消除逐像素水平归约）。稀疏模式把引导计算限制在亮度结构区域（信任掩码
由结构张量能量 + 支撑窗膨胀生成），掩码非激活区输出纯核——是阈值近似
（低对比度内容下与密集最大差 ~0.009），不是逐位一致保证。

VapourSynth 跨帧并行，吞吐随核数近线性扩展（`core.num_threads`）。

SIMD 设计要点：

- 内层全部 FMA + 快速倒数（`rcp` + Newton-Raphson 一步），无 exp/div；
  Lorentzian 各项天然 SIMD 友好；
- U/V 共享全部引导权重，单次计算双平面累加；
- AVX-512 迁移直接：lane 8→16，除 lane 索引常量外无需结构变化；
  `__mmask` 可消掉 sup 对齐 padding；
- 已知优化空间：输出像素维度的 8 路并行（消除逐像素水平归约）、
  引导权重按相位相同的列复用。

---

## 8. 后续研究方向

- **Encoder-D 推断与条件标定（当前最高优先级）**：先在片段/场景级估计 box、
  triangle、bicubic 类退化核，或增加显式 D 参数；随后分别标定 q。现有跨 D
  汇总 AUC 0.777 掩盖了 box/triangle 内约 0.99 与 bicubic 0.66 的巨大差异。
- **Kernel-derived phase**（misalign1 残差的正解）：估计亚色度像素的 Y/C 相位
  偏移并平移吸附/迁移目标（比较相邻相位 polyphase 分量的能量），而非门控。
- **更完整的不可观测模式压力集**：从三个轴/棋盘 Nyquist 扩展到斜向、周期扫频、
  相位扫频及不同实际 D；把“风险衰减”量化为频率响应，不再用单个反例代表零空间。
- **边缘 alpha-profile（动画线稿专项）**：拟合两侧颜色 + 共享覆盖率 α(x) 的
  三段式 ridge 模型。ms 门控已使 hardL_softC 回退 plain（不再输），alpha-profile
  的目标是在该类内容上**赢过** plain；algo6 的法向 1D 迁移是其前置。
- **沿边缘方向的高阶核**：当前 bump 是 Lorentzian；把沿边缘分量换成 Lanczos
  轮廓（保留负瓣锐化）可能进一步减少沿边缘模糊。
- **时域深化**：半像素 ME（奇数相位分集目前的整数限制）、双帧遮挡检测、
  沿运动轨迹的 3–5 帧联合恢复；让奇数位移 tap 在超锐边缘上从"持平"变"收益"
  （当前被 footprint 骑跨测试保守抑制）。
- **数据一致性 BP 的候选滤波器**：D_h 固定 2×2 box 只是原型；按候选编码器
  滤波器（bilinear/bicubic/area）+ siting 联动选择。
- **性能追回**：逐输出像素 SIMD（消逐像素水平归约）、plain 路径恢复快速
  重采样但保持边界语义一致、AVX-512（lane 8→16，`__mmask` 消 padding）。

---

## 9. 文件结构

```
src/lgcr.h          共享声明（结构体、inline 核函数、转换模板）
src/maps.cpp        引导图构建（结构张量 / Lc / mutual-structure 门控）+ BilinAxis
src/kernels.cpp     权重表、可分离/径向重采样、色度轴
src/recon.cpp       reconstructChroma（引导主路径）+ plainChroma
src/algos.cpp       algo4 内部 LGF/selector、sharpenPlane、algo6 仿射映射与细节迁移
src/plugin.cpp      VapourSynth 胶水（Recon + Sharpen + TRecon 注册与调度）
Makefile            make / make liblgcr_scalar.so / make check
test/test_lgcr.py   合成边缘消融测试（平坦区/垂直边/对角边 + AVX2对标量）
test/test_algo6.py  algo6 定向别名、qgate/ms 连续性、缩放及标量一致性断言
test/battery.py     正式评测电池（含错位、三向 Nyquist、线稿、噪声、2×、时域）
test/eval_gates.py  独立获益标签 × 三种实际 D 的门控 ROC/风险覆盖评测
test/eval_image.py  固定源哈希/边缘掩码的真实图像保留算法回归
test/recon_image.py 静态图管线（PNG → 模拟420退化 → 重建 → PNG）
test/selfguide_y.py Y 自引导 oracle 实验（阴性结果，见 §5 第 10 条）
```
