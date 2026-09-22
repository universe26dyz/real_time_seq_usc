#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter, defaultdict
from pathlib import Path


def n_acq(ds):
    for name in ('number_of_acquisitions','getNumberOfAcquisitions'):
        fn=getattr(ds,name,None)
        if fn: return int(fn())
    raise RuntimeError('Cannot find acquisition-count method on ismrmrd.Dataset')


def read_acq(ds,i):
    for name in ('read_acquisition','readAcquisition'):
        fn=getattr(ds,name,None)
        if fn: return fn(i)
    raise RuntimeError('Cannot find acquisition reader on ismrmrd.Dataset')


def detect_group(h5path:Path):
    import h5py
    with h5py.File(h5path,'r') as f:
        tops=list(f.keys())
        if 'dataset' in tops: return 'dataset', tops
        for k in tops:
            try:
                names=set(f[k].keys())
            except Exception:
                continue
            if 'data' in names or 'xml' in names:
                return k, tops
    return tops[0] if tops else 'dataset', tops


def is_flag(acq, flag):
    for name in ('isFlagSet','is_flag_set'):
        fn=getattr(acq,name,None)
        if fn:
            try: return bool(fn(flag))
            except Exception: pass
    return False


def proto_expected(path:Path):
    s=str(path)
    m=re.search(r'(T1[345])_(5|60)',s,re.I)
    if not m: return None,None,None
    fam=m.group(1).upper(); ns=int(m.group(2)); return fam,ns,ns*350


def audit_one(path:Path,outdir:Path):
    import ismrmrd
    fam, expected_slices, expected_imaging = proto_expected(path)
    group,tops=detect_group(path)
    ds=ismrmrd.Dataset(str(path), group, create_if_needed=False)
    total=n_acq(ds)

    flag_names=[
      'ACQ_IS_NOISE_MEASUREMENT','ACQ_IS_PARALLEL_CALIBRATION',
      'ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING','ACQ_IS_NAVIGATION_DATA',
      'ACQ_IS_PHASECORR_DATA','ACQ_IS_DUMMYSCAN_DATA','ACQ_IS_RTFEEDBACK_DATA',
      'ACQ_IS_HPFEEDBACK_DATA','ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA',
      'ACQ_FIRST_IN_SLICE','ACQ_LAST_IN_SLICE','ACQ_LAST_IN_MEASUREMENT']
    flags={n:getattr(ismrmrd,n) for n in flag_names if hasattr(ismrmrd,n)}
    nonimg_exclude={'ACQ_IS_NOISE_MEASUREMENT','ACQ_IS_PARALLEL_CALIBRATION',
                    'ACQ_IS_NAVIGATION_DATA','ACQ_IS_PHASECORR_DATA','ACQ_IS_DUMMYSCAN_DATA',
                    'ACQ_IS_RTFEEDBACK_DATA','ACQ_IS_HPFEEDBACK_DATA',
                    'ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA'}

    rows=[]; imaging=[]
    hist_samples=Counter(); hist_channels=Counter(); hist_flags=Counter(); hist_rep=Counter()
    by_slice=defaultdict(list)

    for i in range(total):
        a=read_acq(ds,i); idx=a.idx
        fset=[n for n,v in flags.items() if is_flag(a,v)]
        for n in fset: hist_flags[n]+=1
        cls='non_imaging' if nonimg_exclude.intersection(fset) else 'imaging_like'
        def iv(x):
            try:return int(x)
            except Exception:return int(x.item())
        r={
          'file_acq_index':i,'class':cls,'scan_counter':iv(a.scan_counter),
          'timestamp':iv(a.acquisition_time_stamp),'samples':iv(a.number_of_samples),
          'channels':iv(a.active_channels),'traj_dim':iv(a.trajectory_dimensions),
          'slice':iv(idx.slice),'lin':iv(idx.kspace_encode_step_1),'rep':iv(idx.repetition),
          'avg':iv(idx.average),'segment':iv(idx.segment),'phase':iv(idx.phase),
          'contrast':iv(idx.contrast),'set':iv(idx.set),'flags':';'.join(fset),
          'pos_x':float(a.position[0]),'pos_y':float(a.position[1]),'pos_z':float(a.position[2]),
          'read_x':float(a.read_dir[0]),'read_y':float(a.read_dir[1]),'read_z':float(a.read_dir[2]),
          'phase_x':float(a.phase_dir[0]),'phase_y':float(a.phase_dir[1]),'phase_z':float(a.phase_dir[2]),
          'slice_x':float(a.slice_dir[0]),'slice_y':float(a.slice_dir[1]),'slice_z':float(a.slice_dir[2]),
        }
        rows.append(r); hist_samples[r['samples']]+=1; hist_channels[r['channels']]+=1; hist_rep[r['rep']]+=1
        if cls=='imaging_like':
            r['imaging_ordinal']=len(imaging); imaging.append(r); by_slice[r['slice']].append(r)

    per_slice={}
    for slc,rr in sorted(by_slice.items()):
        lins=[x['lin'] for x in rr]
        per_slice[str(slc)]={'count':len(rr),'count_is_350':len(rr)==350,
            'first_20_lin':lins[:20],'last_20_lin':lins[-20:],
            'lin_min':min(lins) if lins else None,'lin_max':max(lins) if lins else None,
            'lin_unique_count':len(set(lins))}

    t13=None
    if fam=='T13':
        bad=[]
        for r in imaging:
            exp=r['imaging_ordinal']%144
            if r['lin']!=exp:
                bad.append({'ord':r['imaging_ordinal'],'slice':r['slice'],'lin':r['lin'],'expected':exp})
                if len(bad)>=30: break
        t13={'rule':'LIN == imaging_ordinal % 144','passes':len(bad)==0,'first_mismatches':bad}

    tag=path.parent.name
    outdir.mkdir(parents=True,exist_ok=True)
    csvp=outdir/f'{tag}_acquisitions.csv'
    fields=list(rows[0].keys())+['imaging_ordinal']
    with csvp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            rr=dict(r); rr.setdefault('imaging_ordinal',''); w.writerow(rr)

    # Slice-boundary windows (imaging-like only)
    bp=outdir/f'{tag}_slice_boundaries.csv'; br=[]; prev=None
    for j,r in enumerate(imaging):
        if prev is None or r['slice']!=prev:
            for x in imaging[max(0,j-5):min(len(imaging),j+6)]:
                z=dict(x); z['transition_at_ordinal']=j; br.append(z)
            prev=r['slice']
    if br:
        with bp.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(br[0].keys())); w.writeheader(); w.writerows(br)

    summary={'path':str(path),'size_bytes':path.stat().st_size,'top_level_h5_keys':tops,
      'ismrmrd_group':group,'protocol_family':fam,'expected_slices':expected_slices,
      'expected_imaging_acq':expected_imaging,'total_acq_all_types':total,
      'imaging_like_acq':len(imaging),'imaging_count_matches_expected':(len(imaging)==expected_imaging if expected_imaging else None),
      'unique_imaging_slices':sorted(by_slice),'slice_count_matches_expected':(len(by_slice)==expected_slices if expected_slices else None),
      'all_slices_have_350':all(len(v)==350 for v in by_slice.values()) if by_slice else False,
      'samples_histogram':dict(hist_samples),'channels_histogram':dict(hist_channels),
      'repetition_histogram':dict(hist_rep),'flags_histogram':dict(hist_flags),
      'per_slice':per_slice,'t13_lin_check':t13,
      'acq_csv':str(csvp),'slice_boundary_csv':str(bp) if br else None}
    jp=outdir/f'{tag}_summary.json'; jp.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    files=sorted(args.root.rglob('testdata.h5'))
    if not files: raise SystemExit(f'No testdata.h5 under {args.root}')
    print(f'Found {len(files)} files')
    allsum=[]
    for p in files:
        print('\n',p)
        try:
            s=audit_one(p,args.out); allsum.append(s)
            for k in ['ismrmrd_group','total_acq_all_types','imaging_like_acq','expected_imaging_acq','imaging_count_matches_expected','unique_imaging_slices','slice_count_matches_expected','all_slices_have_350','samples_histogram','channels_histogram','repetition_histogram']:
                print(f'  {k}: {s[k]}')
            if s['t13_lin_check'] is not None: print('  T13 LIN mod144:',s['t13_lin_check']['passes'])
        except Exception as e:
            print('  ERROR:',repr(e)); allsum.append({'path':str(p),'error':repr(e)})
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    (out/'all_h5_summary.json').write_text(json.dumps(allsum,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nSummary:',out/'all_h5_summary.json')

if __name__=='__main__': main()
