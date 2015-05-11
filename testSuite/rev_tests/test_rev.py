import subprocess
import os
import shutil
from optparse import OptionParser

import threading
import datetime

from Command import *

test_path = os.getcwd()
log_name = ''
global_rev = '0'

def get_rev(path):
    prev_path = os.getcwd()
    os.chdir(path)

    p = subprocess.Popen('git svn find-rev HEAD', stdout=subprocess.PIPE, shell=True)
    (rev, err) = p.communicate()
    if rev == '':
        p = subprocess.Popen('git rev-parse HEAD', stdout=subprocess.PIPE, shell=True)
        (rev, err) = p.communicate()
    os.chdir(prev_path)

    print 'Revision', rev

    return rev.strip()

def get_trili_dir():
    if not global_rev.isdigit():
        return os.path.join(os.path.expanduser("~"), 'Trilinos/11.12')
    elif int(global_rev) > 1000:
        return os.path.join(os.path.expanduser("~"), 'Trilinos/11.12')
    else:
        return os.path.join(os.path.expanduser("~"), 'Trilinos/11.2')

def log(text):
    global test_path
    prev_path = os.getcwd()
    os.chdir(test_path)
    f = open(log_name+'.txt', 'a')
    f.write(text)
    f.close()
    os.chdir(prev_path)

def log_exists(name):
    return os.path.isfile(name+'.txt')

def log_set_name(name):
    global log_name
    log_name = name

def platform(testing=False):
    return 'PLAT=%s' % ('cartdefault' if not testing else 'carttesting')

def shared_dir(fredwubs_path):
    return 'SHARED_DIR='+os.path.join(os.path.expanduser("~"), 'testing/')

def trilinos_home():
    return 'TRILINOS_HOME=%s' % get_trili_dir()

def library_path():
    path = os.environ.get('LD_LIBRARY_PATH', '')
    path = (':' + path if path else path)
    return ('LD_LIBRARY_PATH=%s/lib' % get_trili_dir()) + path

def normal_path():
    path = os.environ.get('PATH', '')
    path = (':' + path if path else path)
    return 'PATH='+os.path.join(os.path.expanduser("~"), 'local/bin')+path

def env_vars(fredwubs_path, testing=False):
    envs = {}
    for env in [platform(testing), shared_dir(fredwubs_path), trilinos_home(), library_path(), normal_path()]:
        for i in env.split(' '):
            var = i.split('=')
            if len(var) == 2:
                envs[var[0]] = var[1]
            else:
                print 'Error with environment variable', i
    return envs

def replace_if_not_replaced(text, old, new):
    if (new in text):
        return text

    return text.replace(old, new)

def build(fredwubs_path, path, testing=False, target=None):

    log("Starting HYMLS build process\n")

    prev_path = os.getcwd()
    src = os.path.join(fredwubs_path, path)
    os.chdir(src)

    p = Command('make clean', env=env_vars(fredwubs_path, testing))
    p.run()

    log("Clean output:\n\n")
    log(p.out)

    log("Clean errors:\n\n")
    log(p.err)

    #patch makefile
    makefile_path = os.path.join(src, 'Makefile')
    makefile = open(makefile_path, 'r')
    t = makefile.read()
    makefile.close()

    # fix compilation with git
    t = replace_if_not_replaced(t, 'rev=\\"$(shell svnversion -n)\\"', 'rev="\\"$(shell svnversion -n)\\""')

    # ccache for faster compilation
    t = replace_if_not_replaced(t, 'include Makefile.inc', 'include Makefile.inc\nCXX:=ccache ${CXX}')
 
    #disable openmp support because it's broken with Intel 15
    t = replace_if_not_replaced(t, '${EXTRA_LD_FLAGS}', '${EXTRA_LD_FLAGS}\nCXX_FLAGS:=$(filter-out -openmp,$(CXX_FLAGS))\nCXX_FLAGS:=$(filter-out -fopenmp,$(CXX_FLAGS))')

    makefile = open(makefile_path, 'w')
    makefile.write(t)
    makefile.close()

    #remove symlinks in fvm
    if os.path.lexists('NOX_Epetra_LinearSystem_Belos.H'):
        os.remove('NOX_Epetra_LinearSystem_Belos.H')
    if os.path.lexists('NOX_Epetra_LinearSystem_Belos.C'):
        os.remove('NOX_Epetra_LinearSystem_Belos.C')

    target = (target if target else '')
    p = Command('make -j 20 '+target, env=env_vars(fredwubs_path, testing))
    p.run()

    log("Build output:\n\n")
    log(p.out)

    log("Build errors:\n\n")
    log(p.err)

    os.chdir(prev_path)

def integration_tests(fredwubs_path, procs=1):
    log("Starting integration tests\n")

    prev_path = os.getcwd()
    path = os.path.join(fredwubs_path, 'hymls', 'testSuite', 'integration_tests')
    os.chdir(path)

    p = ParallelCommand('../../src/integration_tests', env_vars(fredwubs_path), procs)
    p.run()

    log("Test output:\n\n")
    log(p.out)

    log("Test errors:\n\n")
    log(p.err)

    os.chdir(prev_path)

def unit_tests(fredwubs_path):
    log("Starting unit tests\n")

    prev_path = os.getcwd()
    path = os.path.join(fredwubs_path, 'hymls', 'testSuite', 'unit_tests')
    os.chdir(path)

    p = ParallelCommand('./main', env=env_vars(fredwubs_path))
    p.run()

    log("Test output:\n\n")
    log(p.out)

    log("Test errors:\n\n")
    log(p.err)

    os.chdir(prev_path)

def fvm_test(fredwubs_path, test_path, testing=False, procs=1):
    log("Starting fvm test\n")

    prev_path = os.getcwd()
    path = os.path.join(test_path, '..')
    os.chdir(path)

    nodes = 1
    size = (32 if procs == 1 else 64)
    ssize = 8
    levels = 3
    re_end = 0

    p = Command('python runtest.py %d %d %d %d %d' % (nodes, procs, size, ssize, levels), env=env_vars(fredwubs_path, testing))
    p.run()

    fname = 'FVM_LDCav_nn%02d_np%d_nx%d_sx%d_L%d_%d' % (nodes, procs, size, ssize, levels, re_end)
    shutil.copy(fname+'.out', test_path+'/'+fname+('' if not testing else '_testing')+'.out')

    log("Test output:\n\n")
    log(p.out)

    log("Test errors:\n\n")
    log(p.err)

    os.chdir(prev_path)

def build_hymls(fredwubs_path, testing=False):
    build(fredwubs_path, 'hymls/src', testing, 'libhymls_'+('cartdefault' if not testing else 'carttesting')+'.a integration_tests')

def build_unit_tests(fredwubs_path, testing=False):
    build(fredwubs_path, 'hymls/testSuite/unit_tests', testing)

def build_fvm(fredwubs_path, testing=False):
    build(fredwubs_path, 'fvm/src', testing)

def run(name, *args):
    # Rerun every test for now and instead append logs, because we run
    # some tests 4 times. Previously I wanted to cancel tests that were
    # already done.

    #~ if log_exists(name):
        #~ return

    log_set_name(name)
    globals()[name](*args)

def set_commit(commit):
    subprocess.call('git reset --hard '+commit, shell=True)

def options():
    parser = OptionParser(usage="python test_rev.py [commit]")
    (options, args) = parser.parse_args()
    if len(args) > 0:
        set_commit(args[0])

def main():
    global test_path, global_rev
    fredwubs_path = os.path.join(os.path.expanduser("~"), 'testing/fredwubs')
    running_path = os.getcwd()
    os.chdir(fredwubs_path)

    print 'Path is', fredwubs_path

    options()

    os.chdir(running_path)
    rev = get_rev(fredwubs_path)
    global_rev = rev
    if not os.path.isdir(rev):
        os.mkdir(rev)
    else:
        # We stop here because the tests were already performed.
        # It would be nice to do this per test, but this doesn't work,
        # see the "run" method
        return

    test_path = os.path.join(running_path, rev)
    os.chdir(test_path)

    run('build_hymls', fredwubs_path)

    run('build_fvm', fredwubs_path)
    run('fvm_test', fredwubs_path, test_path)
    run('fvm_test', fredwubs_path, test_path, False, 16)

    run('build_hymls', fredwubs_path, True)

    run('integration_tests', fredwubs_path)
    run('integration_tests', fredwubs_path, 16)

    if os.path.isfile(os.path.join(fredwubs_path, 'hymls/testSuite/unit_tests/Makefile')):
        run('build_unit_tests', fredwubs_path)
        run('unit_tests', fredwubs_path)

    run('build_fvm', fredwubs_path, True)
    run('fvm_test', fredwubs_path, test_path, True)
    run('fvm_test', fredwubs_path, test_path, True, 16)

if __name__ == '__main__':
    main()
