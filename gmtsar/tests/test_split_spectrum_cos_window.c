// test_split_spectrum_cos_window.c
//
// Regression test for the heap-buffer-underflow fixed in
// gmtsar/split_spectrum.c's cos_window() on 2026-09-24. See
// python/docs/PATHWAY_FORWARD.md for the full write-up.
//
// Root cause: cos_window() writes a bandpass window into filter[], a
// heap array of exactly N doubles, using loops whose start/end indices
// are derived from nc/flat_nb/cos_nb. For some real parameter
// combinations (confirmed on real NISAR_CSAF data, scene NSR_20260331A),
// nc - flat_nb - cos_nb - 1 comes out negative, so a loop starts writing
// at filter[-1], filter[-2], ... -- corrupting the heap allocator's
// bookkeeping for whatever chunk sits just before filter[] in memory.
// That corruption doesn't crash immediately; it only surfaces later as
// "double free or corruption (out)" (SIGABRT) when the corrupted chunk
// is eventually freed, far from cos_window() itself.
//
// This file is intentionally standalone (doesn't #include gmt.h,
// gmtsar.h, or tiffio.h, and doesn't link against libgmt/libtiff) so it
// can be built and run anywhere with a plain C compiler + AddressSanitizer,
// without needing a full GMTSAR build. cos_window()/fliplr()/die() below
// are copied verbatim from split_spectrum.c -- keep them in sync if that
// file's cos_window() changes.
//
// Build & run:
//   gcc -fsanitize=address -g -O0 test_split_spectrum_cos_window.c -o /tmp/tcw -lm
//   /tmp/tcw   # exit 0 and "ALL CASES PASSED" means the fix holds
//
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define PI 3.1415926535897932

void die(const char *s1, const char *s2) {
    fprintf(stderr, "die: %s %s\n", s1, s2);
    exit(1);
}

int fliplr(double *a, int N) {
    int i;
    double d;
    for (i = 0; i < (int)(N / 2); i++) {
        d = a[i];
        a[i] = a[N - 1 - i];
        a[N - 1 - i] = d;
    }
    return (1);
}

// Verbatim copy of the FIXED cos_window() body from split_spectrum.c.
int cos_window(double fc, double fb, double fs, int N, double *filter) {
    int i, nc, nb, flat_nb, cos_nb;
    if (fabs(fc) > fs / 3) die("center frequency too big!", "");
    for (i = 0; i < N; i++) filter[i] = 0.0;
    nc = (int)fabs(round(fc / fs * N));
    nb = (int)round(fb / fs * N);
    flat_nb = (int)round(nb / 4);
    cos_nb = (int)round(nb / 4);

    int i_start;

    i_start = nc - flat_nb - 1;
    if (i_start < 0) i_start = 0;
    for (i = i_start; i < nc + flat_nb && i < N; i++)
        filter[i] = 1;

    i_start = nc - flat_nb - cos_nb - 1;
    if (i_start < 0) i_start = 0;
    for (i = i_start; i < nc - flat_nb - 1 && i < N; i++)
        filter[i] = 0.5 - cos(PI * (i - (nc - flat_nb - cos_nb - 1) + 1) / cos_nb) / 2.0;

    i_start = nc + flat_nb;
    if (i_start < 0) i_start = 0;
    for (i = i_start; i < nc + flat_nb + cos_nb && i < N; i++)
        filter[i] = cos(PI * (i - (nc + flat_nb) + 1) / cos_nb) / 2.0 + 0.5;

    if (fc < 0) fliplr(filter, N);
    return (1);
}

static int check_no_nan_inf(double *filter, int N, const char *label) {
    for (int i = 0; i < N; i++) {
        if (isnan(filter[i]) || isinf(filter[i])) {
            fprintf(stderr, "FAIL %s: filter[%d] is NaN/Inf (%.6f)\n", label, i, filter[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    int N = 65536;
    double fs = 48000000.0;
    double *filter = (double *)malloc(N * sizeof(double));
    int ok = 1;

    // Case 1: the exact reproduction of the real crash -- a tiny
    // bandwidth relative to fs/N drives nc down near 0, which used to
    // underflow filter[] by a few elements before this fix.
    double bc = 50.0;
    cos_window(bc, bc, fs, N, filter);
    ok &= check_no_nan_inf(filter, N, "case1 positive fc (bc=50)");
    cos_window(-bc, bc, fs, N, filter);
    ok &= check_no_nan_inf(filter, N, "case1 negative fc (bc=50)");

    // Case 2: nc == 0 exactly.
    cos_window(0.0, 0.0, fs, N, filter);
    ok &= check_no_nan_inf(filter, N, "case2 fc=0, fb=0");

    // Case 3: a realistic in-bounds NISAR-like sub-band bandwidth --
    // must still produce the expected flat-top value of 1.0 at filter[nc]
    // (regression: the fix must not change already-in-bounds behavior).
    double bc3 = 15000000.0 / 3.0;
    cos_window(bc3, bc3, fs, N, filter);
    ok &= check_no_nan_inf(filter, N, "case3 realistic bc");
    int nc3 = (int)fabs(round(bc3 / fs * N));
    if (fabs(filter[nc3] - 1.0) > 1e-9) {
        fprintf(stderr, "FAIL case3: filter[nc]=%.9f, expected 1.0\n", filter[nc3]);
        ok = 0;
    }

    // Case 4: fc at the fs/3 boundary the die() check allows (largest
    // legal bc) -- nc + flat_nb + cos_nb should clamp cleanly at N if it
    // would otherwise overrun.
    double bc4 = fs / 3.0 - 1.0;
    cos_window(bc4, bc4, fs, N, filter);
    ok &= check_no_nan_inf(filter, N, "case4 fc near fs/3 boundary");

    free(filter);
    if (ok) {
        printf("ALL CASES PASSED\n");
        return 0;
    }
    return 1;
}
