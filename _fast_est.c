
#include <stdlib.h>
#include <string.h>

int fast_octopipe_est(
    const int *assigned, const int *stage_f, const int *stage_b,
    int S, int D, int M)
{
    int *dev_free = (int *)calloc(D, sizeof(int));
    int *dev_last = (int *)malloc(D * sizeof(int));
    for (int i = 0; i < D; i++) dev_last[i] = 2;

    int cap = (M * S + 4) * 4;
    int **dq = (int **)malloc(D * sizeof(int *));
    int *dqn = (int *)calloc(D, sizeof(int));
    for (int d = 0; d < D; d++)
        dq[d] = (int *)malloc(cap * sizeof(int));

    int d0 = assigned[0];
    for (int mid = 0; mid < M; mid++) {
        int n = dqn[d0];
        dq[d0][n]=0; dq[d0][n+1]=0; dq[d0][n+2]=mid; dq[d0][n+3]=0;
        dqn[d0] += 4;
    }

    int done = 0, target = M * S * 2, Sm1 = S - 1;
    long long KS = (long long)(S * M + 1);
    long long KP = 1LL << 55;

    while (done < target) {
        int gd = -1, gi = -1, gs = 0x7fffffff;

        for (int d = 0; d < D; d++) {
            int *q = dq[d];
            int qn = dqn[d];
            if (qn == 0) continue;

            int fr = dev_free[d];
            int pref = (dev_last[d] == 0) ? 1 : 0;

            long long lb = 0x7fffffffffffffffLL;
            int li = -1, ls = 0x7fffffff;

            for (int i = 0; i < qn; i += 4) {
                int rt = q[i], wt = q[i+1], mid = q[i+2], sid = q[i+3];
                int st = (fr >= rt) ? fr : rt;
                long long p = (wt == pref) ? 0 : KP;
                long long tb = (wt == 0) ? (long long)mid * S + sid
                                         : (long long)(Sm1 - sid) * M + mid;
                long long key = p + (long long)st * KS + tb;
                if (key < lb) { lb = key; li = i; ls = st; }
            }
            if (li >= 0 && ls < gs) { gd = d; gi = li; gs = ls; }
        }
        if (gd < 0) break;

        int *q = dq[gd];
        int wt = q[gi+1], mid = q[gi+2], sid = q[gi+3];
        int tail = dqn[gd] - 4;
        if (gi != tail) {
            q[gi]=q[tail]; q[gi+1]=q[tail+1];
            q[gi+2]=q[tail+2]; q[gi+3]=q[tail+3];
        }
        dqn[gd] -= 4;

        int dur = (wt == 0) ? stage_f[sid] : stage_b[sid];
        int end = gs + dur;
        dev_free[gd] = end;
        done++;

        if (wt == 0) {
            dev_last[gd] = 0;
            if (sid < Sm1) {
                int dn = assigned[sid+1]; int n = dqn[dn];
                dq[dn][n]=end; dq[dn][n+1]=0; dq[dn][n+2]=mid; dq[dn][n+3]=sid+1;
                dqn[dn] += 4;
            }
            if (sid == Sm1) {
                int n = dqn[gd];
                q[n]=end; q[n+1]=1; q[n+2]=mid; q[n+3]=sid;
                dqn[gd] += 4;
            }
        } else {
            dev_last[gd] = 1;
            if (sid > 0) {
                int dp = assigned[sid-1]; int n = dqn[dp];
                dq[dp][n]=end; dq[dp][n+1]=1; dq[dp][n+2]=mid; dq[dp][n+3]=sid-1;
                dqn[dp] += 4;
            }
        }
    }

    int result = 0;
    for (int d = 0; d < D; d++) {
        if (dev_free[d] > result) result = dev_free[d];
        free(dq[d]);
    }
    free(dq); free(dqn); free(dev_free); free(dev_last);
    return result;
}
