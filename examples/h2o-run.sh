#!/bin/bash

set -x
#sbatch examples/submit_h2o_opt.sh facebook/opt-1.3b opt 2048 0.125
#sbatch examples/submit_h2o_opt.sh facebook/opt-2.7b opt 2048 0.125
#sbatch examples/submit_h2o_opt.sh facebook/opt-6.7b opt 2048 0.125
#sbatch examples/submit_h2o_opt.sh facebook/opt-13b opt 2048 0.125
#sbatch examples/submit_h2o_opt.sh facebook/opt-30b opt 2048 0.125
#sbatch examples/submit_h2o_opt.sh facebook/opt-1.3b opt 2048 0.25
#sbatch examples/submit_h2o_opt.sh facebook/opt-2.7b opt 2048 0.25
#sbatch examples/submit_h2o_opt.sh facebook/opt-6.7b opt 2048 0.25
#sbatch examples/submit_h2o_opt.sh facebook/opt-13b opt 2048 0.25
#sbatch examples/submit_h2o_opt.sh facebook/opt-30b opt 2048 0.25
sbatch examples/submit_h2o_opt.sh facebook/opt-1.3b opt 2048 0.0625
sbatch examples/submit_h2o_opt.sh facebook/opt-2.7b opt 2048 0.0625
sbatch examples/submit_h2o_opt.sh facebook/opt-6.7b opt 2048 0.0625
sbatch examples/submit_h2o_opt.sh facebook/opt-13b opt 2048 0.0625
sbatch examples/submit_h2o_opt.sh facebook/opt-30b opt 2048 0.0625
set +x

