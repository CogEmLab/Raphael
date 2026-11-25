"""Runs inside CaImAn's own environment (caiman_env/), started by
raphael.imaging.cnmfe; imports nothing from Raphael.

    python caiman_runner.py spec.json

One-photon pipeline, as CaImAn's CNMF-E example and the helping-behaviour
analysis (CogEmLab/helping-behaviour-calcium-imaging) run it:

1. rigid (or piecewise-rigid) motion correction, NoRMCorre, on a
   high-pass-filtered copy of the movie (`gSig_filt`, 1p);
2. the corrected movie as a C-order memmap;
3. CNMF-E: correlation/peak-to-noise initialisation, ring-model background,
   OASIS deconvolution;
4. CaImAn's component evaluation (SNR and spatial correlation; CNN off),
   which sets `idx_components`. Nothing is removed: every component is
   exported with an `accepted` flag, so curation can change later;
5. the detrended trace (`detrend_df_f(detrend_only=True)`, CaImAn's
   recommendation for 1p, where the background is not a usable F0).

Writes to spec['out_dir']: the CaImAn model (`{name}.hdf5`, for CaImAn's
own tools), `{name}_cnmfe.npz` (plain arrays Raphael reads), and
`{name}_summary.json`. Spec keys: movie, out_dir, name, fr, n_processes,
motion (CaImAn motion params), cnmfe (CaImAn init/patch/temporal params),
quality (evaluation thresholds), keep_movies.
"""

import json
import sys
import time
from pathlib import Path


def _setup_cluster(n_processes):
    import caiman as cm

    if n_processes and n_processes > 1:
        _, dview, n = cm.cluster.setup_cluster(backend='multiprocessing',
                                               n_processes=int(n_processes),
                                               ignore_preexisting=True)
        return dview, n
    return None, 1


def main(spec_path):
    import caiman as cm
    import numpy as np
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction import cnmf
    from caiman.source_extraction.cnmf import params as cnmf_params

    spec = json.loads(Path(spec_path).read_text())
    out_dir = Path(spec['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = spec['name']
    movie = str(spec['movie'])
    timing = {}

    dview, n_processes = _setup_cluster(spec.get('n_processes'))
    try:
        motion = dict(spec['motion'])
        opts = cnmf_params.CNMFParams(params_dict={
            'fnames': [movie], 'fr': spec['fr'],
            **{k: (tuple(v) if isinstance(v, list) else v) for k, v in motion.items()},
        })

        start = time.perf_counter()
        mc = MotionCorrect([movie], dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        if opts.get('motion', 'pw_rigid'):
            fname_mc = mc.fname_tot_els
            shifts = np.stack([np.asarray(mc.x_shifts_els).mean(1),
                               np.asarray(mc.y_shifts_els).mean(1)], axis=1)
            border = int(np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                            np.max(np.abs(mc.y_shifts_els)))))
        else:
            fname_mc = mc.fname_tot_rig
            rig = np.asarray(mc.shifts_rig)               # (rows, cols) = (y, x)
            shifts = rig[:, ::-1]                         # -> (x, y)
            border = int(np.ceil(np.max(np.abs(rig))))
        max_shift = border
        if opts.get('motion', 'border_nan') == 'copy':
            border = 0
        fname_new = cm.save_memmap(fname_mc, base_name=str(out_dir / f'memmap_{name}_'),
                                   order='C', border_to_0=border, dview=dview)
        timing['motion_correction_s'] = round(time.perf_counter() - start, 1)

        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        c = dict(spec['cnmfe'])
        gsig = c.pop('gSig')
        gsiz = c.pop('gSiz', None) or [2 * g + 1 for g in gsig]
        q = dict(spec['quality'])
        opts.change_params(params_dict={
            'method_init': 'corr_pnr', 'K': None, 'nb': 0, 'nb_patch': 0,
            'low_rank_background': None, 'normalize_init': False, 'center_psf': True,
            'only_init': True, 'method_deconvolution': 'oasis', 'border_pix': border,
            'gSig': tuple(gsig), 'gSiz': tuple(gsiz),
            **{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()},
            **q,
        })

        start = time.perf_counter()
        model = cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
        model.fit(images)
        timing['cnmfe_s'] = round(time.perf_counter() - start, 1)

        start = time.perf_counter()
        model.estimates.evaluate_components(images, model.params, dview=dview)
        model.estimates.detrend_df_f(detrend_only=True)
        sub = images[::max(T // 1000, 1)]
        correlation, pnr = cm.summary_images.correlation_pnr(sub, gSig=gsig[0], swap_dim=False)
        mean = np.asarray(sub, dtype=np.float32).mean(0)
        timing['evaluation_s'] = round(time.perf_counter() - start, 1)

        est = model.estimates
        A = est.A.tocsc()
        accepted = np.zeros(A.shape[1], bool)
        accepted[np.asarray(est.idx_components, dtype=int)] = True
        model.save(str(out_dir / f'{name}.hdf5'))
        np.savez_compressed(
            out_dir / f'{name}_cnmfe.npz',
            A_data=A.data.astype(np.float32), A_indices=A.indices, A_indptr=A.indptr,
            A_shape=np.asarray(A.shape), dims=np.asarray(dims),
            C=np.asarray(est.C, np.float32), YrA=np.asarray(est.YrA, np.float32),
            S=np.asarray(est.S, np.float32), detrended=np.asarray(est.F_dff, np.float32),
            accepted=accepted, snr=np.asarray(est.SNR_comp, np.float32),
            r_value=np.asarray(est.r_values, np.float32),
            shifts=np.asarray(shifts, np.float32), correlation=np.asarray(correlation, np.float32),
            pnr=np.asarray(pnr, np.float32), mean=mean,
        )
        summary = {
            'caiman': cm.__version__, 'frames': int(T), 'dims': list(map(int, dims)),
            'fr': spec['fr'], 'n_processes': n_processes, 'components': int(A.shape[1]),
            'accepted': int(accepted.sum()), 'max_shift_px': max_shift, **timing,
            'params': json.loads(json.dumps(model.params.to_dict(), default=str)),
        }
        (out_dir / f'{name}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        if dview is not None:
            cm.stop_server(dview=dview)

    if not spec.get('keep_movies', False):
        for f in {*map(str, [fname_mc] if isinstance(fname_mc, str) else fname_mc),
                  str(fname_new)}:
            Path(f).unlink(missing_ok=True)
    return summary


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1]), indent=2)[:2000])
