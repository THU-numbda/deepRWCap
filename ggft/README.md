# General Green's Function Table (GGFT)

The GGFT solver uses finite difference method (FDM) to solve Green's funtions of cubes containing any dielectrics.

It is useful for floating random walk method, since every random-walk step needs to be sampled according to the Green's function of the current cube.

It can also be used to generate training data for a machine learning model.
For example, to generate random dielectric configurations, run

```bash
make -j
bin/ggft <nthreads> <total_count> <output_folder> <poisson/gradient>
```