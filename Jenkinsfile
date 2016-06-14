node("SC-030083L") {
    dir("/localdata/f_buildn/ESSEX_workspace/hymls_pipeline/"){
   stage 'check out from bitbucket'
   git credentialsId: 'da583372-2a2e-4e64-a1a5-cdc431625a46', url: 'https://bitbucket.org/hymls/hymls.git'
   stage 'execute shell script'
   shellPrefix="#!/bin/bash\n"
   sh(shellPrefix + '''# configure modulesystem to load newer bash!
export MODULEPATH=/tools/modulesystem/modulefiles
module() { eval `/usr/bin/modulecmd bash $*`; }

# set ccache path
export CCACHE_DIR=/home_local/f_buildn/ESSEX_workspace/.ccache
export FC="ccache gfortran" CC="ccache gcc" CXX="ccache g++"

# let our ci-build-script handle everything else
./ci-sc-dlr.sh -e gcc-4.9.2-openmpi -t 11.12.1
''')
 }
}
