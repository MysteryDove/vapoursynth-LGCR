#include "lgcr.h"

namespace lgcr {

// LGF / selector / NEDI / sharpen

void buildLGF(const Plane &Y, int cw, int ch, double rw, double rh,
                     double shiftX, double shiftY, const Plane &C, int radius, double eps,
                     Plane &a, Plane &b, Plane &conf, bool cedge) {
    // Luma level per chroma sample = POINT SAMPLE at the sited position (not
    // the footprint average used by the sim path). The regression needs Y at
    // exactly the position C refers to; a misaligned aperture biases the
    // slope by a * offset * slope, which dominates on gradients.
    Plane ls(cw, ch);
    for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx)
            ls.at(cx, cy) = bilinear(Y, (cx + 0.5) * rw - 0.5 + shiftX,
                                        (cy + 0.5) * rh - 0.5 + shiftY);
    for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
            double sY = 0, sY2 = 0, sC = 0, sYC = 0;
            int n = 0;
            for (int j = std::max(0, y - radius); j <= std::min(ch - 1, y + radius); ++j)
                for (int i = std::max(0, x - radius); i <= std::min(cw - 1, x + radius); ++i) {
                    const double ly = ls.at(i, j);
                    const double cc = C.at(i, j);
                    sY += ly; sY2 += ly * ly; sC += cc; sYC += ly * cc;
                    ++n;
                }
            const double meanY = sY / n, meanC = sC / n;
            const double cov = sYC / n - meanY * meanC;
            const double var = sY2 / n - meanY * meanY;
            const double av = cov / (var + eps);
            a.at(x, y) = static_cast<float>(av);
            b.at(x, y) = static_cast<float>(meanC - av * meanY);
            // regression confidence: no luma variance -> no predictive power
            float cf = static_cast<float>(var / (var + eps));
            if (cedge) {
                // same chroma-transition-width fade as the sim path
                float cmin = 1e30f, cmax = -1e30f;
                for (int j = std::max(0, y - radius); j <= std::min(ch - 1, y + radius); ++j)
                    for (int i = std::max(0, x - radius); i <= std::min(cw - 1, x + radius); ++i) {
                        cmin = std::min(cmin, C.at(i, j));
                        cmax = std::max(cmax, C.at(i, j));
                    }
                const float lgx = (ls.at(x + 1, y) - ls.at(x - 1, y)) * 0.5f;
                const float lgy = (ls.at(x, y + 1) - ls.at(x, y - 1)) * 0.5f;
                const float gl = std::hypot(lgx, lgy);
                if (gl > 1e-9f && cmax - cmin > 1e-6f) {
                    const float cgx = (C.at(x + 1, y) - C.at(x - 1, y)) * 0.5f;
                    const float cgy = (C.at(x, y + 1) - C.at(x, y - 1)) * 0.5f;
                    const float gradN = std::fabs(cgx * lgx / gl + cgy * lgy / gl);
                    const float wc = (cmax - cmin) / (gradN + 1e-6f);
                    cf *= std::clamp((2.2f - wc) / 0.7f, 0.0f, 1.0f);
                }
            }
            conf.at(x, y) = cf;
        }
    }
}

// out = (1-lam)*plain + lam*(a*Y0 + b), evaluated at full-res positions
void blendLGF(Plane &outU, Plane &outV,
                     const Plane &aU, const Plane &bU, const Plane &aV, const Plane &bV,
                     const Plane &confU, const Plane &confV, const Plane &Y,
                     const ChromaAxis &ax, const ChromaAxis &ay,
                     int cw, int ch, float lam) {
    const BilinAxis bcx = buildBilinAxis(ax.pos, cw);
    const BilinAxis bcy = buildBilinAxis(ay.pos, ch);
    const BilinAxis blx = buildBilinAxis(ax.lpos, Y.w);
    const BilinAxis bly = buildBilinAxis(ay.lpos, Y.h);
    for (int oy = 0; oy < outU.h; ++oy) {
        float *ru = outU.row(oy);
        float *rv = outV.row(oy);
        const int cyi = bcy.i0[oy], lyi = bly.i0[oy];
        const float cyf = bcy.f[oy], lyf = bly.f[oy];
        for (int ox = 0; ox < outU.w; ++ox) {
            const float L0 = bilinearFast(Y, blx.i0[ox], blx.f[ox], lyi, lyf);
            const int cxi = bcx.i0[ox];
            const float cxf = bcx.f[ox];
            const float au = bilinearFast(aU, cxi, cxf, cyi, cyf);
            const float bu = bilinearFast(bU, cxi, cxf, cyi, cyf);
            const float av = bilinearFast(aV, cxi, cxf, cyi, cyf);
            const float bv = bilinearFast(bV, cxi, cxf, cyi, cyf);
            const float lu = lam * bilinearFast(confU, cxi, cxf, cyi, cyf);
            const float lv = lam * bilinearFast(confV, cxi, cxf, cyi, cyf);
            ru[ox] = (1.0f - lu) * ru[ox] + lu * (au * L0 + bu);
            rv[ox] = (1.0f - lv) * rv[ox] + lv * (av * L0 + bv);
        }
    }
}

// algo=4: per-pixel mechanism selector.
//   out = plain + lam * (w2 * (guided - plain) + w3 * conf * (lgf - plain))
// w2/w3 come from the guided pass (hard-edge-ness x axis/diagonal split).
void blendSelector(Plane &outU, Plane &outV,
                          const Plane &gU, const Plane &gV,
                          const Plane &aU, const Plane &bU, const Plane &aV, const Plane &bV,
                          const Plane &confU, const Plane &confV, const Plane &Y,
                          const ChromaAxis &ax, const ChromaAxis &ay,
                          const Plane &w2, const Plane &w3,
                          int cw, int ch, float lam) {
    const BilinAxis bcx = buildBilinAxis(ax.pos, cw);
    const BilinAxis bcy = buildBilinAxis(ay.pos, ch);
    const BilinAxis blx = buildBilinAxis(ax.lpos, Y.w);
    const BilinAxis bly = buildBilinAxis(ay.lpos, Y.h);
    for (int oy = 0; oy < outU.h; ++oy) {
        float *ru = outU.row(oy);
        float *rv = outV.row(oy);
        const float *gur = gU.row(oy);
        const float *gvr = gV.row(oy);
        const float *w2r = w2.row(oy);
        const float *w3r = w3.row(oy);
        const int cyi = bcy.i0[oy], lyi = bly.i0[oy];
        const float cyf = bcy.f[oy], lyf = bly.f[oy];
        for (int ox = 0; ox < outU.w; ++ox) {
            const float L0 = bilinearFast(Y, blx.i0[ox], blx.f[ox], lyi, lyf);
            const int cxi = bcx.i0[ox];
            const float cxf = bcx.f[ox];
            const float w2v = w2r[ox] * lam;
            const float w3v = w3r[ox] * lam;
            if (w2v < 1e-4f && w3v < 1e-4f)
                continue;
            const float au = bilinearFast(aU, cxi, cxf, cyi, cyf);
            const float bu = bilinearFast(bU, cxi, cxf, cyi, cyf);
            const float av = bilinearFast(aV, cxi, cxf, cyi, cyf);
            const float bv = bilinearFast(bV, cxi, cxf, cyi, cyf);
            const float cu = bilinearFast(confU, cxi, cxf, cyi, cyf);
            const float cv = bilinearFast(confV, cxi, cxf, cyi, cyf);
            ru[ox] += w2v * (gur[ox] - ru[ox]) + w3v * cu * (au * L0 + bu - ru[ox]);
            rv[ox] += w2v * (gvr[ox] - rv[ox]) + w3v * cv * (av * L0 + bv - rv[ox]);
        }
    }
}


static void solve4x4(float R[4][4], float r[4], float a[4]) {
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int row = col + 1; row < 4; ++row)
            if (std::fabs(R[row][col]) > std::fabs(R[piv][col]))
                piv = row;
        if (piv != col) {
            for (int k = col; k < 4; ++k) std::swap(R[piv][k], R[col][k]);
            std::swap(r[piv], r[col]);
        }
        const float d = R[col][col];
        if (std::fabs(d) < 1e-12f)
            continue;
        for (int row = col + 1; row < 4; ++row) {
            const float f = R[row][col] / d;
            for (int k = col; k < 4; ++k) R[row][k] -= f * R[col][k];
            r[row] -= f * r[col];
        }
    }
    for (int row = 3; row >= 0; --row) {
        float s = r[row];
        for (int k = row + 1; k < 4; ++k) s -= R[row][k] * a[k];
        a[row] = (std::fabs(R[row][row]) > 1e-12f) ? s / R[row][row] : 0.25f;
    }
}

static float nediPixel(const Plane &C, float scx, float scy, double eps, double arMargin) {
    const int i0 = static_cast<int>(std::floor(scx - 0.5f));
    const int j0 = static_cast<int>(std::floor(scy - 0.5f));
    // covariance over a 6x6 window of source samples
    float R[4][4] = {}, r[4] = {};
    for (int j = j0 - 2; j <= j0 + 3; ++j)
        for (int i = i0 - 2; i <= i0 + 3; ++i) {
            if (i < 1 || j < 1 || i >= C.w - 1 || j >= C.h - 1)
                continue;
            const float v[4] = { C.at(i - 1, j - 1), C.at(i + 1, j - 1),
                                 C.at(i - 1, j + 1), C.at(i + 1, j + 1) };
            const float yk = C.at(i, j);
            for (int p = 0; p < 4; ++p) {
                for (int q = 0; q < 4; ++q) R[p][q] += v[p] * v[q];
                r[p] += v[p] * yk;
            }
        }
    for (int p = 0; p < 4; ++p) R[p][p] += float(eps);
    float a[4];
    solve4x4(R, r, a);
    const float v[4] = { C.at(i0, j0), C.at(i0 + 1, j0),
                         C.at(i0, j0 + 1), C.at(i0 + 1, j0 + 1) };
    float out = a[0] * v[0] + a[1] * v[1] + a[2] * v[2] + a[3] * v[3];
    if (arMargin >= 0.0) {
        float mn = std::min(std::min(v[0], v[1]), std::min(v[2], v[3]));
        float mx = std::max(std::max(v[0], v[1]), std::max(v[2], v[3]));
        out = std::clamp(out, mn - float(arMargin), mx + float(arMargin));
    }
    return out;
}

void nediChroma(const Plane &U, const Plane &V, Plane &outU, Plane &outV,
                       const ChromaAxis &ax, const ChromaAxis &ay,
                       const Plane &plainU, const Plane &plainV,
                       float lam, double eps, double arMargin) {
    for (int oy = 0; oy < outU.h; ++oy) {
        const float scy = ay.pos[oy];
        for (int ox = 0; ox < outU.w; ++ox) {
            const float scx = ax.pos[ox];
            // coincident with a source sample: passthrough
            const float fx = scx - std::floor(scx);
            const float fy = scy - std::floor(scy);
            if (std::min(fx, 1.0f - fx) < 1e-3f && std::min(fy, 1.0f - fy) < 1e-3f)
                continue; // keep plain (which is the source sample here)
            const float nu = nediPixel(U, scx, scy, eps, arMargin);
            const float nv = nediPixel(V, scx, scy, eps, arMargin);
            float *ru = outU.row(oy);
            float *rv = outV.row(oy);
            ru[ox] = (1.0f - lam) * plainU.row(oy)[ox] + lam * nu;
            rv[ox] = (1.0f - lam) * plainV.row(oy)[ox] + lam * nv;
        }
    }
}


void sharpenPlane(Plane &p, const Plane *guideLc, const Plane &guideY,
                         double rw, double rh, double shiftX, double shiftY,
                         const SharpenData &d) {
    // guideLc: chroma-res luma footprint map (nullptr for luma self-guide)
    const int w = p.w, h = p.h;
    Plane src = p; // copy
    const float invG2 = 1.0f / float(d.gspatial * d.gspatial);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float c0 = src.at(x, y);
            // reference luma at this sample
            float L0;
            if (guideLc) {
                L0 = bilinear(guideY, (x + 0.5) * rw - 0.5 + shiftX,
                                      (y + 0.5) * rh - 0.5 + shiftY);
            } else {
                L0 = c0;
            }
            float maxAbsDL = 0.0f, hullMin = 1e30f, hullMax = -1e30f;
            for (int j = -2; j <= 2; ++j)
                for (int i = -2; i <= 2; ++i) {
                    if (guideLc)
                        maxAbsDL = std::max(maxAbsDL,
                                            std::fabs(guideLc->at(x + i, y + j) - L0));
                    else
                        maxAbsDL = std::max(maxAbsDL,
                                            std::fabs(src.at(x + i, y + j) - L0));
                    hullMin = std::min(hullMin, src.at(x + i, y + j));
                    hullMax = std::max(hullMax, src.at(x + i, y + j));
                }
            // sigma floor from the local first-order slope (median |4-neighbor
            // diff|): on smooth slopes this lets curvature taps pass the gate;
            // on hard steps the median is ~0 and the gate stays strict.
            const float c0l = src.at(x, y);
            float nd[4] = {
                std::fabs(src.at(x + 1, y) - c0l), std::fabs(src.at(x - 1, y) - c0l),
                std::fabs(src.at(x, y + 1) - c0l), std::fabs(src.at(x, y - 1) - c0l),
            };
            std::sort(nd, nd + 4);
            const float slopeMed = (nd[1] + nd[2]) * 0.5f;
            const float sig = std::max(std::max(float(d.sigma), float(d.sratio) * maxAbsDL),
                                       3.0f * slopeMed);
            const float invS2 = 1.0f / (sig * sig);
            float hp = 0.0f, wsum = 0.0f;
            for (int j = -2; j <= 2; ++j)
                for (int i = -2; i <= 2; ++i) {
                    if (i == 0 && j == 0)
                        continue;
                    const float dL = (guideLc ? guideLc->at(x + i, y + j)
                                              : src.at(x + i, y + j)) - L0;
                    const float sim = 1.0f / (1.0f + dL * dL * invS2);
                    const float sp = 1.0f / (1.0f + float(i * i + j * j) * invG2);
                    const float n = sp * sim;
                    hp += n * (c0 - src.at(x + i, y + j));
                    wsum += n;
                }
            float out = c0;
            if (wsum > 1e-6f)
                out = c0 + float(d.alpha) * hp / wsum;
            if (d.arMargin >= 0.0)
                out = std::clamp(out, hullMin - float(d.arMargin), hullMax + float(d.arMargin));
            p.at(x, y) = out;
        }
    }
}

// Back-projection data consistency: one cheap IBP-style iteration.
// residual = C_src - box2x2(C_out);  C_out += bp * residual (per 2x2 block)
void backProject(Plane &cOut, const Plane &cSrc, float bp) {
    const int cw = cSrc.w, ch = cSrc.h;
    for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx) {
            const int x0 = 2 * cx, y0 = 2 * cy;
            const float p00 = cOut.at(x0, y0), p10 = cOut.at(x0 + 1, y0);
            const float p01 = cOut.at(x0, y0 + 1), p11 = cOut.at(x0 + 1, y0 + 1);
            const float r = cSrc.at(cx, cy) - 0.25f * (p00 + p10 + p01 + p11);
            const float dcorr = bp * r;
            cOut.at(x0, y0) += dcorr;
            cOut.at(x0 + 1, y0) += dcorr;
            cOut.at(x0, y0 + 1) += dcorr;
            cOut.at(x0 + 1, y0 + 1) += dcorr;
        }
}

// ---------------------------------------------------------------------------
// TRecon support: integer-pel luma block matching
// ---------------------------------------------------------------------------

void blockMatch(const Plane &cur, const Plane &nbr, int blockSize, int search,
                float tsad, std::vector<int16_t> &mvx, std::vector<int16_t> &mvy,
                std::vector<float> &conf, int &bw, int &bh) {
    bw = (cur.w + blockSize - 1) / blockSize;
    bh = (cur.h + blockSize - 1) / blockSize;
    mvx.assign(size_t(bw) * bh, 0);
    mvy.assign(size_t(bw) * bh, 0);
    conf.assign(size_t(bw) * bh, 0.0f);
    const float invT2 = 1.0f / (tsad * tsad);

    for (int by = 0; by < bh; ++by)
        for (int bx = 0; bx < bw; ++bx) {
            const int x0 = bx * blockSize, y0 = by * blockSize;
            const int bw_ = std::min(blockSize, cur.w - x0);
            const int bh_ = std::min(blockSize, cur.h - y0);
            float best = 1e30f, sad0 = 1e30f;
            double sum = 0.0, sum2 = 0.0;
            int bdx = 0, bdy = 0;
            for (int j = 0; j < bh_; j += 2)
                for (int i = 0; i < bw_; i += 2) {
                    const float v = cur.at(x0 + i, y0 + j);
                    sum += v;
                    sum2 += double(v) * v;
                }
            const float npix = float((bw_ + 1) / 2 * ((bh_ + 1) / 2));
            for (int dy = -search; dy <= search; ++dy)
                for (int dx = -search; dx <= search; ++dx) {
                    if (x0 + dx < 0 || y0 + dy < 0 ||
                        x0 + dx + bw_ > cur.w || y0 + dy + bh_ > cur.h)
                        continue;
                    float sad = 0;
                    for (int j = 0; j < bh_; j += 2)
                        for (int i = 0; i < bw_; i += 2)
                            sad += std::fabs(cur.at(x0 + i, y0 + j) -
                                             nbr.at(x0 + dx + i, y0 + dy + j));
                    if (dx == 0 && dy == 0)
                        sad0 = sad;
                    // Equal SAD: prefer the smaller motion (zero-motion
                    // bias). On a straight edge the tangential component is
                    // unobservable (aperture problem) but harmless — the
                    // fetched samples are identical along iso-contours.
                    const int mnorm = std::abs(dx) + std::abs(dy);
                    const int bnorm = std::abs(bdx) + std::abs(bdy);
                    if (sad < best || (sad == best && mnorm < bnorm)) {
                        best = sad;
                        bdx = dx;
                        bdy = dy;
                    }
                }
            // Zero-motion snap: accept a nonzero vector only when it beats
            // staying put by a real margin. In noisy flat blocks the search
            // otherwise mines a random small vector out of the noise.
            if ((bdx != 0 || bdy != 0) && sad0 - best <= 0.10f * tsad * npix) {
                bdx = 0;
                bdy = 0;
                best = sad0;
            }
            const size_t idx = size_t(by) * bw + bx;
            mvx[idx] = int16_t(bdx);
            mvy[idx] = int16_t(bdy);
            const float sadPx = best / npix;
            // match quality x observability. Motion is only observable where
            // the LUMA block has texture: a flat block matches everywhere
            // (margin-based uniqueness fails differently: it also fires on
            // straight edges via the aperture ambiguity). The earlier
            // first-scan tie rule picked a far corner at FULL confidence and
            // dragged chroma texture across the frame (0.012 -> 0.059).
            const float var = float(std::max(0.0, sum2 / npix - (sum / npix) * (sum / npix)));
            const float matchQ = 1.0f / (1.0f + sadPx * sadPx * invT2);
            const float uniq = var / (var + 0.25f * tsad * tsad);
            conf[idx] = matchQ * uniq;
        }
}

} // namespace lgcr
