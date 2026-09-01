"""Collect manuscript numbers (v14 layout: S_t / T / G / G_adj, A1/A2/B3/D/N) from a results dir.
usage: python src/paper_numbers.py results/paper5c17   -> <dir>/PAPER_NUMBERS2.json + table rows printed
"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
P = Path(sys.argv[1]); out = {}
def f2(x): return f"{x:.2f}".replace("0.", ".").replace("-.", "$-$.")
def f2s(x):  # signed
    s = f"{x:+.2f}".replace("0.", ".").replace("+.", "$+$.").replace("-.", "$-$.");  return s
def m(x): return f"{x*1000:+.1f}".replace("+", "$+$").replace("-", "$-$")
SSL = ['wavlm','hubert','xlsr','w2v2lv60','w2vbert']
LAB = {'wavlm':'WavLM-Large','hubert':'HuBERT-Large','xlsr':'XLS-R 300M','w2v2lv60':'wav2vec2-LV60','w2vbert':'w2v-BERT 2.0'}
LAB3 = {'wavlm':'WavLM-L','hubert':'HuBERT-L','xlsr':'XLS-R','w2v2lv60':'w2v2-LV60','w2vbert':'w2v-BERT~2.0'}
# ---------- Table 1
rows=[]; t1={}
for s in SSL:
    t=pd.read_csv(P/f'cloning_summary_{s}.csv'); k=t.iloc[0]; tt=t.iloc[1]
    t1[s]={'kway':[k.spkdisjoint_macro_f1,k.ci_lo,k.ci_hi],'tts':[tt.spkdisjoint_macro_f1,tt.ci_lo,tt.ci_hi],'layers':[k.chosen_layers,tt.chosen_layers],
           'ari':k.get('kmeans_ari_L0'),'var_sys':k.get('var_between_system_L0'),'var_spk':k.get('var_between_speaker_L0')}
    rows.append((s,f"{f2(k.spkdisjoint_macro_f1)} [{f2(k.ci_lo)},{f2(k.ci_hi)}]",f"{f2(tt.spkdisjoint_macro_f1)} [{f2(tt.ci_lo)},{f2(tt.ci_hi)}]"))
best=max(SSL,key=lambda s:t1[s]['kway'][0]); T1=[]
for s,kk,ttt in rows:
    if s==best: kk=kk.replace(f2(t1[s]['kway'][0]),"\\textbf{"+f2(t1[s]['kway'][0])+"}",1); ttt=ttt.replace(f2(t1[s]['tts'][0]),"\\textbf{"+f2(t1[s]['tts'][0])+"}",1)
    T1.append(f"{LAB[s]} & {kk} & {ttt} \\\\")
T1.append("\\midrule"); base={}
for nm,lab2,key in [('mfcc_cloning.csv','MFCC(40)$+$stats','mfcc'),('simplecues.csv','Prosodic cues','pros'),('asr_feature_baseline.csv','Whisper-error stats','whi')]:
    d=pd.read_csv(P/nm); a=d[d.setting.astype(str).str.contains('way')].iloc[0]; b=d[d.setting.astype(str).str.contains('ttsonly')].iloc[0]
    base[key]=[a.macro_f1,b.macro_f1]; T1.append(f"{lab2} & {f2(a.macro_f1)} [{f2(a.ci_lo)},{f2(a.ci_hi)}] & {f2(b.macro_f1)} [{f2(b.ci_lo)},{f2(b.ci_hi)}] \\\\")
out['T1']="\n".join(T1); out['t1']=t1; out['base']=base; out['best']=best
# per-class recall, LOO
pc={}
for s in SSL:
    d=pd.read_csv(P/f'cloning_ttsonly_perclass_{s}.csv'); pc[s]={r.iloc[0]:round(float(r['recall']),2) for _,r in d.iterrows()}
out['perclass_recall']=pc
kway=[f for f in glob.glob(str(P/'cloning_*way_layers_wavlm.csv'))][0].split('cloning_')[1].split('_layers')[0]; loo={}
for s in SSL:
    t=pd.read_csv(P/f'cloning_{kway}_layers_{s}.csv'); i=t.loo_acc.idxmax(); tt=pd.read_csv(P/f'cloning_ttsonly_layers_{s}.csv')
    loo[s]={'best':round(t.loo_acc.max(),3),'layer':int(t.loc[i,'layer']),'deep_loo_mean':round(tt.query('17<=layer<=21').loo_acc.mean(),3)}
out['loo']=loo
# ---------- Table 3 (bands) with point estimate of the difference
d=pd.read_csv(P/'bands_spkdisjoint.csv'); T3=[]; bands={}
for s in SSL:
    r=d[d.ssl==s].iloc[0]; bands[s]={k:(round(float(v),3) if isinstance(v,(float,int,np.floating)) else v) for k,v in r.items()}
    diff=r.early_f1-r.deep_f1
    T3.append(f"{LAB3[s]} & {f2(r.early_f1)} [{f2(r.early_lo)},{f2(r.early_hi)}] & {f2(r.deep_f1)} [{f2(r.deep_lo)},{f2(r.deep_hi)}] & {f2s(diff)} [{f2(r.earlyminusdeep_lo)},{f2(r.earlyminusdeep_hi)}] \\\\")
out['T3']="\n".join(T3); out['bands']=bands
out['bands_alt']=pd.read_csv(P/'bands_alt.csv').round(3).to_dict('records') if (P/'bands_alt.csv').exists() else None
out['loo_artifact']=pd.read_csv(P/'loo_artifact.csv').round(3).to_dict('records') if (P/'loo_artifact.csv').exists() else None
# ---------- ASR
out['asr_summary']=pd.read_csv(P/'asr_summary.csv').round(3).to_dict('records'); out['asr_perfect']=pd.read_csv(P/'asr_perfect_deepband.csv').round(3).to_dict('records')
# ---------- Table 2 interventions (WavLM L0) with S_t / T / G / G_adj
iv=pd.read_csv(P/'intervention_wavlm.csv'); iv0=iv[iv.layer==0].set_index('probe'); c2=pd.read_csv(P/'centroid2_wavlm_L0.csv').set_index('probe')
spec=[("resynth_vocos","f5tts","Vocos$\\to$F5"),("resynth_glvocos","f5tts","GL (Vocos mel)$\\to$F5"),("resynth_griffinlim","f5tts","GL (generic)$\\to$F5"),
      ("resynth_hift3","cosyvoice3","HiFT$\\to$C3"),("resynth_s3vc3","cosyvoice3","Token RT$\\to$C3"),("resynth_bigvgan","indextts","BigVGAN$\\to$Index")]
T2=[]; ivd={}
def g(r,c): return None if c not in r.index or pd.isna(r[c]) else float(r[c])
for p,tgt,lab4 in spec:
    if p not in iv0.index: continue
    r=iv0.loc[p]; cr=c2.loc[p] if p in c2.index else None; cc=c2.loc[f"{p}_vs_unmatched"] if f"{p}_vs_unmatched" in c2.index else None
    ivd[p]={'target':tgt,'dP':[g(r,'delta_target'),g(r,'delta_ci_lo'),g(r,'delta_ci_hi')],'S':[g(r,'S_target'),g(r,'S_ci_lo'),g(r,'S_ci_hi')],'top':r.top_class,'aligned':bool(r.target_aligned) if not pd.isna(r.target_aligned) else None,
            'dP_all':{c.replace('dP_',''):round(r[c],3) for c in iv0.columns if c.startswith('dP_')},
            'T':None if cr is None else [g(cr,'T'),g(cr,'T_lo'),g(cr,'T_hi')],'G':None if cr is None else [g(cr,'G'),g(cr,'G_lo'),g(cr,'G_hi')],
            'G_adj':None if cc is None else [g(cc,'G_adj'),g(cc,'G_adj_lo'),g(cc,'G_adj_hi')]}
    dp=f"{f2s(r.delta_target)} [{f2(r.delta_ci_lo)},{f2(r.delta_ci_hi)}]"; S=f"{f2s(r.S_target)} [{f2(r.S_ci_lo)},{f2(r.S_ci_hi)}]"
    T=m(cr['T']) if cr is not None else "--"; G=m(cr['G']) if cr is not None else "--"
    Ga=f"{m(cc['G_adj'])} [{m(cc['G_adj_lo'])},{m(cc['G_adj_hi'])}]" if cc is not None else "--"
    T2.append(f"{lab4} & {dp} & {S} & {T} & {G} & {Ga} \\\\")
# off-target controls under the F5 / C3 targets (CIs from the 2026-08-24 amendment)
for tgt,tl in [("f5tts","F5"),("cosyvoice3","C3")]:
    for p,lab in [("resynth_encodec","EnCodec"),("resynth_dac","DAC"),("resynth_bigvgan","BigVGAN")]:
        if p not in iv0.index: continue
        r=iv0.loc[p]
        u=c2.loc[f"{p}_under_{tgt}"] if f"{p}_under_{tgt}" in c2.index else None
        ivd[f"{p}_under_{tgt}"]={'dP':[g(r,f'dP_{tgt}'),g(r,f'dP_{tgt}_lo'),g(r,f'dP_{tgt}_hi')],'S':[g(r,f'S_{tgt}'),g(r,f'S_{tgt}_lo'),g(r,f'S_{tgt}_hi')],
            'T':None if u is None else [g(u,'T'),g(u,'T_lo'),g(u,'T_hi')],'G':None if u is None else [g(u,'G'),g(u,'G_lo'),g(u,'G_hi')],'top':r.top_class}
        dp=f"{f2s(r[f'dP_{tgt}'])} [{f2(r[f'dP_{tgt}_lo'])},{f2(r[f'dP_{tgt}_hi'])}]" if f'dP_{tgt}_lo' in r.index and not pd.isna(r[f'dP_{tgt}_lo']) else f2s(r[f'dP_{tgt}'])
        S=f"{f2s(r[f'S_{tgt}'])} [{f2(r[f'S_{tgt}_lo'])},{f2(r[f'S_{tgt}_hi'])}]" if f'S_{tgt}_lo' in r.index and not pd.isna(r[f'S_{tgt}_lo']) else f2s(r[f'S_{tgt}'])
        T2.append(f"{lab} (ctrl)$\\to${tl} & {dp} & {S} & {m(u['T']) if u is not None else '--'} & {m(u['G']) if u is not None else '--'} & -- \\\\")
# strongest-control statistic G_min per matched probe
gmin={}
for p in ['resynth_vocos','resynth_hift3','resynth_s3vc3']:
    k=f"{p}_vs_strongest"
    if k in c2.index:
        u=c2.loc[k]; gmin[p]=[g(u,'G_min'),g(u,'G_min_lo'),g(u,'G_min_hi'),u.get('controls')]
out['G_min']=gmin
out['T2']="\n".join(T2); out['interv_L0']=ivd
# w2v-BERT L19 summary
if (P/'intervention_w2vbert.csv').exists():
    iv=pd.read_csv(P/'intervention_w2vbert.csv'); iv19=iv[iv.layer==19].set_index('probe'); c19=pd.read_csv(P/'centroid2_w2vbert_L19.csv').set_index('probe'); l19={}
    for p in iv19.index:
        if p=='clean_real': continue
        r=iv19.loc[p]; cr=c19.loc[p] if p in c19.index else None; cc=c19.loc[f"{p}_vs_unmatched"] if f"{p}_vs_unmatched" in c19.index else None
        l19[p]={'dP':[g(r,'delta_target'),g(r,'delta_ci_lo'),g(r,'delta_ci_hi')],'S':[g(r,'S_target'),g(r,'S_ci_lo'),g(r,'S_ci_hi')],'top':r.top_class,
                'T':None if cr is None else g(cr,'T'),'G':None if cr is None else [g(cr,'G'),g(cr,'G_lo'),g(cr,'G_hi')],'G_adj':None if cc is None else [g(cc,'G_adj'),g(cc,'G_adj_lo'),g(cc,'G_adj_hi')]}
    out['interv_L19_w2vbert']=l19
# layer-wise dP target for matched probes (for text): best layer and L0..L5
ivw=pd.read_csv(P/'intervention_wavlm.csv'); lw={}
for p in ['resynth_vocos','resynth_hift3','resynth_s3vc3','resynth_griffinlim','resynth_glvocos']:
    d=ivw[ivw.probe==p]; lw[p]={int(r.layer):round(float(r.delta_target),3) for _,r in d.iterrows() if not pd.isna(r.delta_target)}
out['interv_layers_wavlm']=lw
# ---------- decoder-only
out['deconly']=pd.read_csv(P/'deconly_wavlm.csv').round(3).to_dict('records')
# ---------- A1 deep probe
if (P/'deep_probe.csv').exists():
    d=pd.read_csv(P/'deep_probe.csv'); out['deep_probe']=d.round(3).to_dict('records')
    piv=d.pivot_table(index=['ssl','band'],columns='classifier',values='macro_f1').round(3); out['deep_probe_pivot']={f"{a}_{b}":v for (a,b),v in piv.to_dict('index').items()}
# ---------- A2 knn mechanism
if (P/'knn_mechanism.csv').exists():
    d=pd.read_csv(P/'knn_mechanism.csv'); out['knn_mech']=d.round(3).to_dict('records')
# ---------- B3
for s in ['wavlm','w2vbert']:
    if (P/f'robustB3_{s}.csv').exists():
        d=pd.read_csv(P/f'robustB3_{s}.csv'); out[f'robustB3_{s}']=d.round(3).to_dict('records')
        if s=='wavlm':
            LABP={'common':'Trim$+$RMS$+$16-bit','common_sym':'$+$16/24/16 resample','mp3_64k':'MP3 64~kbps','lp4k':'Low-pass 4~kHz','hp2k':'High-pass 2~kHz','noise20':'Noise 20~dB','phaserand':'Phase randomization'}
            T4=[]
            for _,r in d.iterrows():
                T4.append(f"{LABP.get(r.pert,r.pert)} & {f2(r.clean_to_clean)} & {f2(r.perturbed_to_perturbed)} & {f2(r.clean_to_perturbed)} & {f2s(r.matched_minus_mismatched)} [{f2(r.diff_ci_lo)},{f2(r.diff_ci_hi)}] \\\\")
            out['T4']="\n".join(T4)
# ---------- N
if (P/'normalization_control.csv').exists(): out['norm']=pd.read_csv(P/'normalization_control.csv').round(3).to_dict('records')
# ---------- D
for nm in ['seeds2_transfer','seeds2_geometry','seeds2_stochasticity']:
    if (P/f'{nm}.csv').exists(): out[nm]=pd.read_csv(P/f'{nm}.csv').round(4).to_dict('records')
json.dump(out, open(P/'PAPER_NUMBERS2.json','w'), indent=1, default=str)
print("wrote", P/'PAPER_NUMBERS2.json')
for k in ['T1','T2','T3','T4']:
    if k in out: print(f"== {k}\n{out[k]}")
