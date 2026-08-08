# LGCR — Luma-Guided Chroma Reconstruction

一个研究型 VapourSynth 4 插件：用**亮度引导的方向自适应卷积核**重建/缩放色度，
解决 YUV 子像素彩边（紫边/绿边/色度锯齿）问题。CPU 实现，AVX2 FMA 快速路径，
全部引导项为有理函数（无 exp/div），对 AVX-512 友好。

当前形态（v1.6）：

```
W'uv = Sim(ΔL) · ( W_kernel + rescue(φ) · Bump_aniso(θ) )
```

- 基核可选 bilinear / bicubic / lanczos / spline16 / spline36 / **jinc**（全 2D 径向路径）
- 边缘方向 θ 与各向异性来自**亮度结构张量**（非裸 Sobel）
- 5 种重建算法入口（`algo=1..5`）+ 独立的边缘感知锐化函数 `lgcr.Sharpen`
  + 时域重建函数 `lgcr.TRecon`（运动补偿多帧相位分集）
- v1.6 语义修正：连续 strength、逻辑 tap 掩码、signed wsum 防护、按核族的
  rescue 门控、H.273 `_ChromaLocation` 支持、稀疏引导（plain 全帧 + 仅结构
  区域引导，1080p→4K 从 1253ms 降到 127ms）、可选 back-projection 一致性项

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
out = core.lgcr.Recon(src, algo=3, reg=0.01)  # LGF 回归路线（文献基线）
out = core.lgcr.Recon(src, algo=5)   # NEDI-lite（仅同尺寸；研究基线，不推荐实用）
out = core.lgcr.Recon(src, algo=1)   # v1.2 历史版本对照
```

**E. 时域重建（动画/多帧素材）**

```python
out = core.lgcr.TRecon(src420, trad=1)            # 前后各 1 帧
out = core.lgcr.TRecon(src420, trad=2, tsearch=8) # 更大窗口/搜索范围
```

机制：亮度块匹配（16×16，整数像素，SAD）把邻帧色度样本对齐到当前帧坐标，
作为额外 tap 注入引导权重循环。**相位分集**：偶数亮度像素位移使 420 采样
相位翻转，邻帧样本携带真正的新信息（t_move2 测试 0.0061 vs 单帧 0.0159，
2.6× 于 Recon）；静态区自动退化为时域平均降噪（t_static_noise 最佳）。
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
| `taps` | 3 | lanczos/jinc 的瓣数 |
| `algo` | 2 | 重建算法：`1`=v1.2 历史版；`2`=v1.3 引导选择（默认）；`3`=LGF 局部线性回归；`4`=逐像素选择器；`5`=NEDI-lite（仅同尺寸） |
| `strength` | 0.8 | 引导/修正强度 λ；**0 = 纯核**（A/B 对照基准，走快速路径） |
| `sigma` | 0.01 | 亮度相似度 σ 下限（归一化单位），相当于噪声地板 |
| `sratio` | 0.15 | 自适应拐点：`sig ≥ sratio · 窗内亮度range` |
| `sdb` | 3.0 (algo≥2) / 1.5 (algo=1) | 斜坡时的 σ 倍数（×窗内range）；显式设置可覆盖任一 algo |
| `stretch` | 1.0 | 沿边缘支撑拉伸倍数（EWA 各向异性，0 = 各向同性） |
| `gsigma` | 2.5 | 引导 bump 宽度（亮度像素），跨边缘方向 |
| `ridge` | 1 | 细线稿（亮度 ridge）检测开关：检出时回退纯核（algo 2/4） |
| `cedge` | 0 | **实验性**：色度过渡宽度检测。实测估计量不稳定（详见 §5），默认关闭 |
| `ar` | 0.0 | anti-ringing：输出 clamp 到窗内色度 hull + 此余量；<0 关闭 |
| `reg` | 0.005 | algo=3/5 的正则化，ε = reg²（也用作 LGF 置信度 conf = var/(var+ε)） |
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
| `strength`/`sigma`/`sratio`/`sdb`/`gsigma`/`stretch`/`ar`/`ridge` | 同 Recon | 引导参数，语义同 Recon（algo=2 路径） |
| `sparse` | 0 | TRecon 默认密集（静态区的时域平均降噪是收益的一部分） |

时域 tap 权重：`w = tconf · sim(ΔL邻帧) · bump(对齐偏差) · 相位新颖度`。
静态块（|mv|≤1 且 tconf>0.5）新颖度=1（纯平均）；运动块仅奇数亮度像素
偏差（相位新颖）的样本得到权重——冗余样本只会稀释带符号基核，被抑制。

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
fade    = ridge检测淡化（细线稿回退纯核）
sim_i  ← 1 − fade·(1 − sim_i)；  rescue ← fade · rescue
W'_i    = sim_i · (W_kernel,i + rescue·bump_i)
C_out   = Σ W'_i·C_i / Σ W'_i                          # 归一化保直流
C_out  ← clamp(C_out, hull_min − ar, hull_max + ar)    # anti-ringing
```

其中 `Lc_i` 是第 i 个色度采样点在其 **sited footprint** 上的亮度均值（420 left
siting 时 footprint 为对应 2×2 亮度块，支持 `loc=left/center`）。U/V 共享全部
引导权重，单次计算双平面累加。

### 3.2 各 algo 一览

| algo | 机制 | 一句话 |
|---|---|---|
| 1 | v1.2：sim 选择 + db 图斜率下限 | 历史对照（对角台阶判别失效、无 ridge 保护） |
| 2 | v1.3+ST：sim 选择 + 法向探针判别 + ridge 淡化 | **默认**，真实内容最稳 |
| 3 | LGF：C ≈ a·Y+b 局部回归（guided filter 路线） | 对角硬边最佳；真实内容有结构迁移代价 |
| 4 | 逐像素选择器：硬边程度 × 轴/对角 → 路由 algo2/algo3/纯核 | 合成/屏幕内容双优 |
| 5 | NEDI-lite：4-tap 协方差自适应（数据自引导） | 研究基线，全线落败（见 §5） |

### 3.3 Sharpen（gated Laplacian）

`HP = Σ n_i(x₀−x_i)/Σ n_i`（`n_i = 空间bump × sim`），`out = x₀ + α·HP`，
hull clamp。门控拉普拉斯：边缘处 HP≈0（无 halo 的来源），软过渡带肩部收紧。
σ 含一阶斜率地板（`max(sigma, sratio·range, 3·slope_med)`）——斜坡不被误当作边缘。

---

## 4. 评测结果

### 可复现评测电池（test/battery.py，v1.6 语义修正后）

`python3 test/battery.py [--all]`，结果写入 `test/results/latest.md`。
当前成绩（jinc3，strength=0.8；注意 v1.6 起 strength 是连续的，0.8 ≠ 全强度，
硬边最大吸附在 1.0 处，hard_v 可达 0.0182）：

| case | plain | algo2 | 判定 |
|---|---|---|---|
| hard_v（严格共边硬边） | 0.0363 | **0.0236** | 赢 |
| hard_d45（对角硬边） | 0.0881 | **0.0797** | 赢 |
| isoluminant（无亮度边色边） | 0.0363 | 0.0363 | 中性（无信息即不动） |
| misalign4（Y/C 错位 4px） | 0.0203 | **0.0188** | 微赢 |
| hardL_softC（硬亮度边+软色度） | **0.0056** | 0.0152 | 输（已知残差，见 §5-7） |
| ridge_line（线稿） | 0.0018 | 0.0018 | 中性（ridge 淡化生效） |
| ramp | 0.0051 | 0.0051 | 中性 |
| texture | 0.0120 | 0.0120 | 中性 |
| noise（σ=0.008） | 0.0375 | 0.0376 | 中性 |
| upscale2x | 0.0386 | **0.0337** | 赢（源空间引导） |
| temporal（边缘移动 1px） | 0.0000 | 0.0000 | 静态区零闪烁 |

时域用例（plain / algo2 / **TRecon**）：

| case | plain | algo2 | TRecon |
|---|---|---|---|
| t_move2（边缘 ±2px/帧，相位分集） | 0.0304 | 0.0159 | **0.0061** |
| t_static_noise（静态+噪声） | 0.0323 | 0.0285 | **0.0241** |

v1.5 及以前的 5 算法对照表见 git 历史 / §5。**strength 连续性验证**：
1e-6 与 plain 完全一致（0.03630），0.2/0.5/0.8/1.0 单调过渡。

### 真实图像（动画截图 1920×1080：大面积天空渐变 + 细线稿）

| | plain | algo1 | algo2 | algo3 | algo4 | algo5 |
|---|---|---|---|---|---|---|
| 全帧 | **0.373** | 0.410 | 0.379 | 0.649 | 0.392 | 0.688 |
| 边缘带 | **1.227** | 1.623 | 1.289 | 3.372 | 1.379 | 2.810 |
| 平滑带 | **0.331** | 0.350 | 0.335 | 0.517 | 0.344 | 0.584 |

解读：这张图 95% 是平滑渐变，几乎没有可恢复空间，引导类的收益集中在硬色边
素材（合成硬边 +67%/+38%）。algo2/algo4 的设计目标是"**不该动的地方不动**"——
真实图像上与 plain 的差距压到了 1.6%（algo2）/ 5%（algo4）。

### f32 输入的影响

plain/algo2 微幅改善（0.349→0.333，少一次 8bit 舍入）；algo3 无改善——
其结构迁移误差源于 C-Y 关系的非线性本身，与量化无关。插件内部全程 f32 计算。

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
   噪声太大——对角硬边法向梯度被系统性低估、中央差分在混合样本上低估梯度，
   两种估法都误伤。正确判别可能需要多尺度或时域信息。
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

**结论**：现有文献几乎都把亮度当作相似度引导，没有工作把高质量插值核
（Lanczos/Spline/Jinc）在亮度重建中暴露的几何信息——边缘方向 θ、边缘强度、
亚像素相位 φ——显式反馈进色度卷积核的形状。GALOSH 的 EWA jinc 最接近，
但 luma 只进 bilateral 相似度项，jinc 核本身不做方向性形变。

LGCR 在文献坐标系中的位置：**JBU 的三个推广**——① 空间基核允许负瓣
（spline/jinc 而非纯高斯），所以能反卷积而不是只会模糊；② 空间核由引导图的
结构张量做 EWA 各向异性变形；③ 相位相关的 rescue 处理上采样网格特有的
"重合 tap 被污染"奇点。等价视角：每输出像素一个即时构造的、引导调制的、
归一化的 transposed convolution（与 dynamic filter network 同思想，但权重
生成器是解析公式而非 CNN）。

---

## 7. 性能与 SIMD

单线程性能（本机 AVX2，1080p420 → 4K444）：

| 路径 | 时间 |
|---|---|
| 纯核（strength=0） | ~79 ms/帧 |
| algo=2 引导 sparse=1（默认） | **~127 ms/帧** |
| algo=2 引导 sparse=0（密集） | ~1253 ms/帧 |

稀疏模式把引导计算限制在亮度结构区域（信任掩码由结构张量能量 + 支撑窗
膨胀生成），与密集模式输出在掩码激活区逐位一致、其余区域等于纯核。

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

- **Kernel-derived phase**：当前 φ 仅几何计算。更激进的形式是从 Lanczos 在 Y
  上的局部响应直接提取相位误差（比较相邻相位 polyphase 分量的能量），检测
  chroma 与 luma 的亚像素失配并做相位补偿。
- **沿边缘方向的高阶核**：当前 bump 是 Lorentzian；把沿边缘分量换成 Lanczos
  轮廓（保留负瓣锐化）可能进一步减少沿边缘模糊。
- **两阶段 NEDI + siting 校正**：algo=5 的正经实现。
- ~~时域~~（v1.6 已实现 TRecon：块匹配对齐 + 相位分集融合）。下一步：
  半像素 ME、双帧遮挡检测、沿运动轨迹的 3–5 帧联合恢复。
- **线稿两翼的正确判别**（多尺度/时域）：见 §5 第 7 条。

---

## 9. 文件结构

```
src/lgcr.h          共享声明（结构体、inline 核函数、转换模板）
src/maps.cpp        引导图构建（结构张量 / db / Lc）+ BilinAxis
src/kernels.cpp     权重表、可分离/径向重采样、色度轴
src/recon.cpp       reconstructChroma（引导主路径）+ plainChroma
src/algos.cpp       LGF / selector blend / NEDI-lite / sharpenPlane
src/plugin.cpp      VapourSynth 胶水（Recon + Sharpen 注册与调度）
Makefile            make / make liblgcr_scalar.so（正确性交叉验证用）
test/test_lgcr.py   合成边缘消融测试（平坦区/垂直边/对角边 + AVX2对标量）
test/battery.py     正式评测电池（11 场景含错位/等亮度/线稿/噪声/2×/时域）
test/recon_image.py 静态图管线（PNG → 模拟420退化 → 重建 → PNG）
test/selfguide_y.py Y 自引导 oracle 实验（阴性结果，见 §5 第 10 条）
```
