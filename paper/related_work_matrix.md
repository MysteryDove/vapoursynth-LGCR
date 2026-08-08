# Related-Work Capability Matrix

`Partial` means the capability is present under a different acquisition model
or is mentioned but not isolated experimentally. Blank cells are avoided;
`not established` means the reviewed source did not support the claim.

| Work | Post-decode existing bitstream | Addresses subsampling | Addresses quantization | Luma guide | Chroma self-guide | Encoder/degrader controlled | Explicit siting intervention | Cel/animation evidence | Separate bleed/alias/phase/ring metrics |
|---|---|---|---|---|---|---|---|---|---|
| Coudoux et al. 2004 | Yes | Partial | Yes, DCT | Edge/block cues | Partial | No | No | No | No |
| Coudoux et al. 2005 | Yes | Yes, 4:1:1 | Yes | Partial | Partial | No | No | No | No |
| Catorina et al. 2007 | Yes | Partial | Yes, DCT | Sobel edge cue | Chroma distance | No | No | No | No |
| Punchihewa 2008 studies | Measurement study | Yes | Yes | No | No | Experimental codec | No | Synthetic color pattern | Color-bleeding measures, not this four-way split |
| Li et al. 2011 | Yes | Partial | Yes, DCT | not established | not established | No | No | No | Chroma PSNR and subjective examples; no four-way split |
| Wada et al. 2015 EJBF | Yes | Yes, 4:2:0 | Yes | Yes | Yes, Cb and Cr | No | No | Three artificial images motivated by cel animation; no released-animation corpus | PSNR and subjective examples; no phase/alias separation |
| Korhonen 2015 | Paired pipeline | Yes | No | Yes | No | Yes | No | No | No |
| Wang et al. 2016 | Paired pipeline | Yes | Partial | Yes | Partial | Yes | No | Screen content, not animation | No |
| Vermeir et al. 2016 | Paired coding tool | Yes | Partial | Yes | No | Yes | No | Screen content, not animation | No |
| Chung et al. 2017 | Paired pipeline | Yes | Partial | Yes | Partial | Yes | No | Screen content, not animation | No |
| Chung et al. 2019 | Paired/content-aware | Yes | Partial | Yes | Yes | Yes | No | Screen content, not animation | No |
| Fu et al. 2019 | Paired HDR/WCG pipeline | Yes | Partial | Yes | Partial | Yes | No | No | No |
| GALOSH RAW 2026 | Partial: RAW/CFA front-end | Half-rate CFA chroma | Denoising rather than codec quantization | Yes | No | Known CFA transform | CFA phase, not H.273 intervention | No | No |
| This report / LGCR | Yes | Yes, 4:2:0 | Not in the primary experiment | Yes | Yes, as gates/statistics | No; candidate kernels only | Yes | Synthetic animation-style scenes; real corpus incomplete | Yes, operational synthetic measures |

The matrix rules out three broad novelty statements: post-decode color-
bleeding repair is not new, luma-guided chroma reconstruction is not new, and
cel-like artificial evaluation is not new. The narrower combination studied
here is unknown-degradation decoder reconstruction plus explicit siting and
guide-mismatch interventions under a reproducible animation-style protocol.
