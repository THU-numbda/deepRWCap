make -j
# generate 100k samples with 12 threads
# (reproducibility of this random generation depends on the thread count used)
bin/ggft 12 100000 dataset poisson
bin/ggft 12 100000 dataset gradient