#include "resample.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chatterbox {

namespace {

int gcd(int a, int b) {
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
}

// Modified Bessel function of the first kind, order 0. Computed via the
// standard series I_0(x) = sum_{k=0}^∞ (x^2 / 4)^k / (k!)^2.
double bessel_i0(double x) {
    double half_x_sq = (x * x) / 4.0;
    double term = 1.0;
    double sum  = 1.0;
    for (int k = 1; k < 50; ++k) {
        term *= half_x_sq / (k * k);
        sum  += term;
        if (term < 1e-15 * sum) break;
    }
    return sum;
}

// firwin(numtaps, cutoff, window=('kaiser', beta)) for low-pass filter.
// Mirrors scipy.signal.firwin defaults: pass_zero=True (LPF), scale=True
// (DC gain normalized to 1).
//
//   h[n] = w_kaiser(n; beta) * cutoff * sinc(cutoff * (n - center))
//
// then scaled so sum(h) == 1.
std::vector<double> firwin_kaiser_lp(int numtaps, double cutoff, double beta) {
    // cutoff is the normalized cutoff in cycles per sample (i.e.,
    // ranges 0..1 where 1 = sample-rate). For a "Nyquist normalized"
    // cutoff of 1/N (= 1 / max(up, down) in resample_poly), this is
    // the half-bandwidth fraction of the Nyquist rate.
    std::vector<double> h(numtaps);
    const double center = (numtaps - 1) / 2.0;
    const double denom_i0_beta = bessel_i0(beta);
    for (int n = 0; n < numtaps; ++n) {
        const double m = n - center;
        // Ideal LPF impulse response: cutoff * sinc_normalized(cutoff * m).
        // sinc_normalized(0) = 1, so at m=0 the value is `cutoff` not `1`.
        // The simplification sin(pi*cutoff*m)/(pi*m) only holds for m != 0.
        const double s = (m == 0.0)
            ? cutoff
            : std::sin(M_PI * cutoff * m) / (M_PI * m);
        // Kaiser window
        const double xrel = m / center;     // -1..1
        const double bessel_arg = beta * std::sqrt(std::max(0.0, 1.0 - xrel * xrel));
        const double w = bessel_i0(bessel_arg) / denom_i0_beta;
        h[n] = w * s;
    }
    // Normalize: scipy's scale=True sets sum(h) = 1.
    double sum = 0.0;
    for (double x : h) sum += x;
    if (sum != 0.0) {
        const double inv = 1.0 / sum;
        for (double& x : h) x *= inv;
    }
    return h;
}

// upfirdn: insert (up-1) zeros between samples, FIR filter `h`,
// downsample by `down`. Standard polyphase formulation:
//   out[n] = sum_{k in 0..L-1, (n*down - k) % up == 0, j = (n*down - k)/up in 0..N-1}
//             h[k] * x[j]
std::vector<float> upfirdn(const std::vector<double>& h,
                             const std::vector<float>& x,
                             int up, int down) {
    const int L  = static_cast<int>(h.size());
    const int xn = static_cast<int>(x.size());
    // Following scipy.signal.upfirdn, output length is
    //   ceil(((xn - 1) * up + L) / down)
    const int out_len = ((xn - 1) * up + L + down - 1) / down;
    std::vector<float> out(out_len, 0.0f);
    for (int n = 0; n < out_len; ++n) {
        const long long m = static_cast<long long>(n) * down;
        // We want k = m - j*up with k in [0, L) and j in [0, xn).
        const int j_max = static_cast<int>(std::min<long long>(xn - 1, m / up));
        const int j_min = static_cast<int>(std::max<long long>(0,
            (m - (L - 1) + up - 1) / up));
        double s = 0.0;
        for (int j = j_min; j <= j_max; ++j) {
            const long long k = m - static_cast<long long>(j) * up;
            if (k >= 0 && k < L) s += h[k] * x[j];
        }
        out[n] = static_cast<float>(s);
    }
    return out;
}

}  // namespace

// Matches scipy.signal.upfirdn output length:
//   output_len = ((n_in - 1) * up + n_h) // down + 1
int upfirdn_output_len(int n_h, int n_in, int up, int down) {
    return ((static_cast<long long>(n_in - 1) * up + n_h - 1) / down) + 1;
}

std::vector<float> resample_audio(const std::vector<float>& in,
                                     int in_sr, int out_sr) {
    if (in.empty() || in_sr <= 0 || out_sr <= 0) return {};
    if (in_sr == out_sr) return in;

    const int g = gcd(in_sr, out_sr);
    const int up   = out_sr / g;
    const int down = in_sr  / g;
    const int max_ud = std::max(up, down);

    // scipy.signal.resample_poly defaults:
    //   half_len = 10 * max(up, down)
    //   numtaps  = 2 * half_len + 1
    //   cutoff   = 1.0 / max(up, down)
    //   window   = ('kaiser', 5.0)  -- our reference uses beta=8.6
    const int half_len = 10 * max_ud;
    const int numtaps  = 2 * half_len + 1;
    const double cutoff = 1.0 / max_ud;
    const double beta   = 8.6;
    auto h = firwin_kaiser_lp(numtaps, cutoff, beta);
    for (auto& v : h) v *= up;

    // scipy pads the FILTER with leading zeros so that
    //   len(h) % down == 0  after pad — this aligns the polyphase
    // phase used for the first input sample.
    const int n_pre_pad = (down - half_len % down) % down;
    // Then chooses n_post_pad such that the upfirdn output is long
    // enough to fit n_pre_remove + n_out samples.
    const int n_in = static_cast<int>(in.size());
    const int n_out = static_cast<int>(
        (static_cast<long long>(n_in) * up + down - 1) / down);
    const int n_pre_remove = (half_len + n_pre_pad) / down;
    int n_post_pad = 0;
    while (upfirdn_output_len(numtaps + n_pre_pad + n_post_pad,
                                  n_in, up, down) < n_out + n_pre_remove) {
        ++n_post_pad;
    }
    // Build padded filter.
    std::vector<double> h_pad;
    h_pad.reserve(numtaps + n_pre_pad + n_post_pad);
    h_pad.insert(h_pad.end(), n_pre_pad, 0.0);
    h_pad.insert(h_pad.end(), h.begin(), h.end());
    h_pad.insert(h_pad.end(), n_post_pad, 0.0);

    auto y = upfirdn(h_pad, in, up, down);
    if (static_cast<int>(y.size()) < n_pre_remove + n_out) {
        std::fprintf(stderr, "resample_audio: y too short (%zu < %d)\n",
                      y.size(), n_pre_remove + n_out);
        return {};
    }
    std::vector<float> out(y.begin() + n_pre_remove,
                              y.begin() + n_pre_remove + n_out);
    return out;
}

}  // namespace chatterbox
