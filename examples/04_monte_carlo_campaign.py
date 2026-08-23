#!/usr/bin/env python3
"""Success rate across random realisations.

Both source reviews make the same point: a stochastic mesher should be judged by
how often it succeeds across seeds, not by one attractive screenshot. This runs
a small campaign and reports the distribution.

Scale up --seeds for a real study; 20 seeds takes about a minute.
"""

from __future__ import annotations

import argparse

from hexaform.io import report as report_io
from hexaform.pipeline.config import DomainConfig, InclusionConfig, OutputConfig, RunConfig
from hexaform.pipeline.run import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--mesh-density", type=int, default=28)
    p.add_argument("--out-dir", default="output/examples/campaign")
    args = p.parse_args()

    reports = []
    for seed in range(args.seeds):
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=args.mesh_density),
            inclusions=InclusionConfig(
                n_particles=(8, 16),        # random count per realisation
                radius_mean=0.07, radius_std=0.018,
            ),
            output=OutputConfig(path=f"{args.out_dir}/mesh_{seed:04d}.e"),
            seed=seed, verbose=False,
        )
        result = run(config)
        reports.append(report_io.build_report(
            config=config, mesh=result.mesh, packing=result.packing,
            validation=result.validation, timings=result.timings,
        ))
        v = result.validation
        print(f"seed {seed:>3}: {'ok  ' if v.passed else 'FAIL'} "
              f"{result.packing.count:>3} particles  "
              f"{result.mesh.n_elements:>8} elems  "
              f"minSJ={v.quality.min_scaled_jacobian:.4f}  "
              f"volerr={v.geometry['relative_volume_error']:.4f}")

    csv_path = report_io.write_campaign_csv(f"{args.out_dir}/campaign.csv", reports)
    print("\n--- summary ---")
    for key, value in report_io.campaign_summary(reports).items():
        print(f"  {key:<28} {value}")
    print(f"  csv                          {csv_path}")


if __name__ == "__main__":
    main()
