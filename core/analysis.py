# core/analysis.py
import numpy as np

SR = 48000
MIN_HNR      = 0.60
MIN_GAIN     = 0.08
MAX_JITTER   = 0.10
STD_ALARM_F0 = 12.0
MAX_CV_VTL   = 0.08
F0_WT        = 0.30
C = 34300.0  # cm/s

def trim(x, r=0.02):
    thr = r*(np.max(np.abs(x))+1e-9)
    idx = np.where(np.abs(x)>thr)[0]
    return x if idx.size==0 else x[idx[0]:idx[-1]+1]

def frame_sig(x, L, H):
    if len(x)<L: return np.empty((0, L))
    frs=[x[s:s+L] for s in range(0, len(x)-L+1, H)]
    return np.stack(frs) if frs else np.empty((0, L))

def preemph(x, a=0.97):
    return np.append(x[0], x[1:]-a*x[:-1])

def parabola(y, i):
    if i<=0 or i>=len(y)-1: return i,y[i]
    y0,y1,y2 = y[i-1],y[i],y[i+1]
    a = y0-2*y1+y2
    if abs(a)<1e-12: return i,y1
    d = (y0-y2)/(2*a)
    return i+d, y1 - 0.25*(y0-y2)*d

def hnr_like(fr):
    ac = np.correlate(fr, fr, 'full')[len(fr)-1:]
    if ac[0]<=1e-9: return 0.0
    ac = ac/(ac[0]+1e-12)
    return float(np.max(ac[1:]))

def spectral_peak(fr, lo, hi):
    from scipy.signal import welch
    f,P = welch(fr, fs=SR, nperseg=min(len(fr), 2048))
    m=(f>=lo)&(f<=hi)
    if not np.any(m): return np.nan
    p=P[m]
    if len(p)>=7: p=np.convolve(p,np.ones(7)/7,mode='same')
    return float(f[m][np.argmax(p)])

def lpc_coeffs(fr, order):
    r=np.correlate(fr,fr,'full')[len(fr)-1:]
    R=r[:order+1]
    a=np.zeros(order+1); a[0]=1.0
    e=R[0]+1e-8
    for i in range(1,order+1):
        acc=R[i]
        for j in range(1,i): acc+=a[j]*R[i-j]
        k=-acc/e
        a_new=a.copy()
        a_new[1:i] = a[1:i] + k*a[i-1:0:-1]
        a_new[i]=k
        a=a_new
        e*=1-k*k
        if e<1e-8: break
    return a,e

def f0_band(profile): return (60,300) if profile=="male" else (120,500)

def estimate_f0_series(x, profile, frame_ms=40, hop_ms=10):
    fmin,fmax=f0_band(profile)
    x=trim(x)
    if len(x)<int(SR*0.05): return np.array([np.nan])
    x=x-np.mean(x); m=np.max(np.abs(x))
    if m>0: x/=m
    L=int(SR*frame_ms/1000.0); H=int(SR*hop_ms/1000.0)
    frames=frame_sig(x,L,H)
    if frames.size==0: return np.array([np.nan])
    win=np.hanning(L)
    min_lag=max(2,int(SR/fmax)); max_lag=min(L-2,int(SR/fmin))
    out=[]
    for fr in frames:
        fr=fr*win
        ac=np.correlate(fr,fr,'full')[L-1:]
        if ac[0]<=0: out.append(np.nan); continue
        ac=ac/(ac[0]+1e-12)
        if max_lag<=min_lag: out.append(np.nan); continue
        seg=ac[min_lag:max_lag]
        if seg.size==0: out.append(np.nan); continue
        i_rel=int(np.argmax(seg)); i=i_rel+min_lag
        i_hat,_=parabola(ac,i)
        if i_hat<=0: out.append(np.nan); continue
        f_raw=SR/i_hat
        def score(f):
            lag=SR/max(f,1e-6); j=int(round(lag))
            return ac[j] if 1<=j<len(ac) else -1.0
        cand=[(f_raw,score(f_raw))]
        if f_raw/2>=fmin: cand.append((f_raw/2,score(f_raw/2)))
        if f_raw*2<=fmax: cand.append((f_raw*2,score(f_raw*2)))
        f0=max(cand,key=lambda z:z[1])[0]
        out.append(f0 if fmin<=f0<=fmax else np.nan)
    return np.array(out,float)

def choose_gender(f0rough):
    if not np.isfinite(f0rough): return "female"
    return "female" if f0rough>=150 else "male"

def vtl_frames_i(y):
    from scipy.signal import welch  # used inside spectral_peak
    y=trim(y)
    if len(y)<int(SR*0.06): return np.array([])
    if np.max(np.abs(y))<MIN_GAIN: return np.array([])
    L=int(SR*0.03); H=int(SR*0.01)
    frames=frame_sig(y,L,H)
    if frames.size==0: return np.array([])
    win=np.hanning(L)
    vtls=[]
    for fr in frames:
        frw=fr*win
        if hnr_like(frw)<MIN_HNR: vtls.append(np.nan); continue
        frw=preemph(frw)
        F1s=[]; F2s=[]
        f1w=spectral_peak(frw,200,450)
        f2w=spectral_peak(frw,1800,3000)
        if np.isfinite(f1w): F1s.append(f1w)
        if np.isfinite(f2w): F2s.append(f2w)
        for order in (12,14,16):
            a,_=lpc_coeffs(frw,order)
            roots=np.roots(a); roots=roots[np.imag(roots)>0.01]
            freqs=np.angle(roots)*(SR/(2*np.pi))
            mags=np.abs(roots)
            bw=-(SR/np.pi)*np.log(mags+1e-8)
            cand=[(f,b) for (f,b) in zip(freqs,bw) if 80<f<4500 and 20<b<500]
            cand.sort(key=lambda z:z[0])
            if cand:
                f1c=cand[0][0]
                if 180<=f1c<=500: F1s.append(f1c)
                for f,b in cand[1:4]:
                    if 1500<=f<=3200:
                        F2s.append(f); break
        if not F1s and not F2s: vtls.append(np.nan); continue
        Lc=[]
        if len(F1s)>0:
            L1=0.60*(C/(4*np.median(F1s)))
            if 13<=L1<=22: Lc.append(L1)
        if len(F2s)>0:
            L2=1.52*(3*C/(4*np.median(F2s)))
            if 13<=L2<=22: Lc.append(L2)
        vtls.append(np.median(Lc) if Lc else np.nan)
    vtls=np.array(vtls,float)
    for i in range(1,len(vtls)):
        if np.isfinite(vtls[i]) and np.isfinite(vtls[i-1]):
            if abs(vtls[i]-vtls[i-1])>MAX_JITTER*max(vtls[i-1],1e-6):
                vtls[i]=np.nan
    return vtls

def robust_vtl(y):
    vt=vtl_frames_i(y)
    vt=vt[np.isfinite(vt)]
    if vt.size<5:
        return np.nan, np.inf, 0
    med=np.median(vt)
    mad=np.median(np.abs(vt-med))+1e-9
    keep=vt[np.abs(vt-med)<=2.5*mad]
    if keep.size<5: return np.nan, np.inf, keep.size
    cv=np.std(keep)/np.mean(keep)
    return float(np.median(keep)), float(cv), keep.size

def height_from(f0, vtl, prof, known_height_cm=None):
    def h_vtl(L,prof):
        if not np.isfinite(L): return np.nan
        return (6.3*L+64) if prof=="female" else (6.8*L+70)
    base=159 if prof=="female" else 171
    h_f0 = base -0.04*(f0-(220 if prof=="female" else 120)) if np.isfinite(f0) else np.nan
    h_v  = h_vtl(vtl,prof)
    if np.isfinite(h_v) and np.isfinite(h_f0):
        h=(1-F0_WT)*h_v + F0_WT*h_f0
    elif np.isfinite(h_v):
        h=h_v
    else:
        h=h_f0
    h=float(np.clip(h,145,200))
    if known_height_cm and np.isfinite(h):
        h=h+0.6*(known_height_cm-h)
    return h

def analyze_wave(y, sr, profile="auto", known_height_cm=None):
    """y: mono float32/64 ndarray, sr: samplerate"""
    # 1) 48000Hzに合わせる（必要なら）
    if sr != SR:
        from scipy.signal import resample_poly
        # 近い整数でresample（品質重視：polyphase）
        y = resample_poly(y, SR, sr)
    # 正規化（安全）
    y = y.astype(np.float64)
    y = y - np.mean(y)
    m = np.max(np.abs(y)) + 1e-12
    y = np.clip(y / m, -1.0, 1.0)

    # 2) 粗F0で性別推定
    tmp = estimate_f0_series(y, "male")
    rough = float(np.nanmedian(tmp)) if np.isfinite(np.nanmedian(tmp)) else np.nan
    prof = choose_gender(rough) if profile=="auto" else profile

    # 3) F0
    f0_series = estimate_f0_series(y, prof)
    voiced = f0_series[np.isfinite(f0_series)]
    f0_med = float(np.nanmedian(voiced)) if voiced.size>0 else np.nan
    f0_std = float(np.nanstd(voiced)) if voiced.size>0 else np.nan

    # 4) VTL
    vtl, cv, n_keep = robust_vtl(y)

    # 5) アラーム
    alarms = []
    if np.isfinite(f0_std) and f0_std > STD_ALARM_F0:
        alarms.append({"type":"F0_STD","msg":f"F0の標準偏差が大きい（σ={f0_std:.1f} > {STD_ALARM_F0} Hz）"})
    if np.isfinite(vtl) and cv > MAX_CV_VTL:
        alarms.append({"type":"VTL_CV","msg":f"VTLが不安定（CV={cv:.3f} > {MAX_CV_VTL}）"})

    # 6) 身長推定
    h = height_from(f0_med, vtl, prof, known_height_cm)

    return {
        "profile": prof,
        "f0_median_hz": None if not np.isfinite(f0_med) else round(f0_med,1),
        "f0_std_hz": None if not np.isfinite(f0_std) else round(f0_std,1),
        "vtl_cm": None if not np.isfinite(vtl) else round(vtl,2),
        "vtl_cv": None if not np.isfinite(vtl) else round(cv,3),
        "vtl_n_keep": int(n_keep),
        "height_cm": None if not np.isfinite(h) else round(h,1),
        "alarms": alarms
    }
