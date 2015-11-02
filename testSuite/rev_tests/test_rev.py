import os
import shutil
from optparse import OptionParser

from Command import Command, ParallelCommand, git_command

test_path = os.getcwd()
log_name = ''
global_rev = '0'

class memoize(dict):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result

def get_rev(path):
    prev_path = os.getcwd()
    os.chdir(path)

    rev = git_command('rev-list --count HEAD')

    os.chdir(prev_path)

    print 'Revision', rev

    return rev.strip()

@memoize
def rev_to_hash(path, rev):
    '''Returns the hash of the nth commit'''
    if not rev.isdigit():
        return rev

    prev_path = os.getcwd()
    os.chdir(path)

    # Get the total amount of commits
    num = git_command('rev-list --count HEAD')
    rev = git_command('log --format="%H" -1 --skip=' + str(int(num)-int(rev)) + ' HEAD')

    os.chdir(prev_path)

    print 'Hash', rev

    return rev.strip()

@memoize
def rev_to_svn_rev(path, rev):
    prev_path = os.getcwd()
    os.chdir(path)

    h = rev_to_hash(path, rev)

    rev = git_command('svn find-rev ' + h)

    os.chdir(prev_path)

    print 'svn revision', rev

    return rev.strip()

def get_trili_dir():
    rev = rev_to_svn_rev(test_path, global_rev)
    if not rev.isdigit():
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

def shared_dir(base_path):
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
    return 'PATH='+os.path.join(os.path.expanduser("~"), 'local/bin') + ':' + get_trili_dir() + path

def env_vars(base_path, testing=False):
    envs = {}
    for env in [platform(testing), shared_dir(base_path), trilinos_home(), library_path(), normal_path()]:
        for i in env.split(' '):
            var = i.split('=')
            if len(var) == 2:
                envs[var[0]] = var[1]
            else:
                print 'Error with environment variable', i
    return envs

def cmake_env_vars(base_path):
    envs = {}
    envs['CXX'] = 'ccache mpicxx'
    path = os.environ.get('PATH', '')
    path = (':' + path if path else path)
    envs['PATH'] = os.path.join(base_path, 'install', 'bin') + ':' + get_trili_dir() + path
    envs['TRILINOS_HOME'] = get_trili_dir()
    return envs

def replace_if_not_replaced(text, old, new):
    if (new in text):
        return text

    return text.replace(old, new)

def build(base_path, path, testing=False):
    log("Starting HYMLS build process\n")

    prev_path = os.getcwd()
    os.chdir(os.path.join(base_path, path))

    #Make a build dir
    if os.path.exists('build'):
        shutil.rmtree('build')
    os.mkdir('build')
    os.chdir('build')

    #Run cmake
    install_path = os.path.join(base_path, 'install')
    command = 'cmake -DCMAKE_INSTALL_PREFIX="' + install_path + '"'
    if not testing:
        command += ' -DCMAKE_BUILD_TYPE=Release'
    p = Command(command + ' .. ', env=cmake_env_vars(base_path))
    p.run()

    log("Cmake output:\n\n")
    log(p.out)

    log("Cmake errors:\n\n")
    log(p.err)

    p = Command('make -j 20', env=cmake_env_vars(base_path))
    p.run()

    log("Build output:\n\n")
    log(p.out)

    log("Build errors:\n\n")
    log(p.err)

    p = Command('make install', env=cmake_env_vars(base_path))
    p.run()

    log("Make install output:\n\n")
    log(p.out)

    log("Make install errors:\n\n")
    log(p.err)

    os.chdir(prev_path)

def build_old(base_path, path, testing=False, target=None):

    log("Starting HYMLS build process\n")

    prev_path = os.getcwd()
    src = os.path.join(base_path, path)
    os.chdir(src)

    p = Command('make clean', env=env_vars(base_path, testing))
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
    p = Command('make -j 20 '+target, env=env_vars(base_path, testing))
    p.run()

    log("Build output:\n\n")
    log(p.out)

    log("Build errors:\n\n")
    log(p.err)

    os.chdir(prev_path)

def integration_tests(base_path, procs=1):
    log("Starting integration tests\n")

    prev_path = os.getcwd()
    path = os.path.join(base_path, 'hymls', 'testSuite', 'integration_tests')
    os.chdir(path)

    exe = os.path.join(base_path, 'hymls', 'build', 'src', 'hymls_integration_tests')
    if not os.path.exists(exe):
        exe = os.path.join(base_path, 'hymls', 'src', 'integration_tests')

    p = ParallelCommand(exe, env_vars(base_path), procs)
    p.run()

    log("Test output:\n\n")
    log(p.out)

    log("Test errors:\n\n")
    log(p.err)

    os.chdir(prev_path)

def unit_tests(base_path):
    log("Starting unit tests\n")

    exe = os.path.join(base_path, 'hymls', 'build', 'testSuite', 'unit_tests', 'unit_tests')
    if not os.path.exists(exe):
        exe = os.path.join(base_path, 'hymls', 'testSuite', 'unit_tests', 'main')

    p = ParallelCommand(exe, env=env_vars(base_path))
    p.run()

    log("Test output:\n\n")
    log(p.out)

    log("Test errors:\n\n")
    log(p.err)

def fvm_test(base_path, test_path, testing=False, procs=1):
    log("Starting fvm test\n")

    prev_path = os.getcwd()
    path = os.path.join(test_path, '..')
    os.chdir(path)

    nodes = 1
    size = (32 if procs == 1 else 64)
    ssize = 8
    levels = 3
    re_end = 0

    p = Command('python runtest.py %d %d %d %d %d' % (nodes, procs, size, ssize, levels), env=env_vars(base_path, testing))
    p.run()

    fname = 'FVM_LDCav_nn%02d_np%d_nx%d_sx%d_L%d_%d' % (nodes, procs, size, ssize, levels, re_end)
    shutil.copy(fname+'.out', test_path+'/'+fname+('' if not testing else '_testing')+'.out')

    log("Test output:\n\n")
    log(p.out)

    log("Test errors:\n\n")
    log(p.err)

    os.chdir(prev_path)

def build_hymls(base_path, testing=False, enable_cmake=True):
    if enable_cmake:
        build(base_path, 'hymls', testing)
    else:
        build_old(base_path, 'fredwubs/hymls/src', testing, 'libhymls_'+('cartdefault' if not testing else 'carttesting')+'.a integration_tests')

def build_unit_tests(base_path, testing=False, enable_cmake=True):
    if not enable_cmake:
        build(base_path, 'hymls/testSuite/unit_tests', testing)

def build_fvm(base_path, testing=False, enable_cmake=True):
    if enable_cmake:
        build(base_path, 'fredwubs/fvm/src', testing)
    else:
        build_old(base_path, 'fredwubs/fvm/src', testing)

def run(name, *args):
    # Rerun every test for now and instead append logs, because we run
    # some tests 4 times. Previously I wanted to cancel tests that were
    # already done.

    #~ if log_exists(name):
        #~ return

    log_set_name(name)
    globals()[name](*args)

def set_commit(commit):
    git_command('reset --hard '+commit)

def sync_commits(hymls_path, fvm_path, commit):
    '''Sync the hymls and fvm versions'''
    prev_path = os.getcwd()

    # Get the hymls commit time
    os.chdir(hymls_path)

    t = git_command('log --format="%at" -1 ' + commit)

    #Set the fvm commit
    os.chdir(fvm_path)
    git_command('fetch', True)
    git_command('svn fetch', True)
    h = git_command('log -1 --format="%H" --before=' + t.strip())
    git_command('reset --hard ' + h)
    os.chdir(prev_path)

def options():
    parser = OptionParser(usage="python test_rev.py [commit]")
    (options, args) = parser.parse_args()
    if len(args) > 0:
        set_commit(args[0])

def main():
    global test_path, global_rev
    base_path = os.path.join(os.path.expanduser("~"), 'testing')
    hymls_path = os.path.join(base_path, 'hymls')
    fvm_path = os.path.join(base_path, 'fredwubs')

    running_path = os.getcwd()
    os.chdir(hymls_path)

    print 'Path is', base_path

    options()

    os.chdir(running_path)
    rev = get_rev(hymls_path)
    global_rev = rev
    if rev and not os.path.isdir(rev):
        os.mkdir(rev)
    else:
        # We stop here because the tests were already performed.
        # It would be nice to do this per test, but this doesn't work,
        # see the "run" method
        return

    sync_commits(hymls_path, fvm_path, rev_to_hash(hymls_path, rev))

    test_path = os.path.join(running_path, rev)
    os.chdir(test_path)

    enable_cmake = False
    if rev_to_svn_rev(hymls_path, rev) == '':
        enable_cmake = True

    run('build_hymls', base_path, False, enable_cmake)

    run('build_fvm', base_path, False, enable_cmake)
    run('fvm_test', base_path, test_path)
    run('fvm_test', base_path, test_path, False, 16)

    run('build_hymls', base_path, True, enable_cmake)

    run('integration_tests', base_path)
    run('integration_tests', base_path, 16)

    if os.path.isfile(os.path.join(base_path, 'hymls/testSuite/unit_tests/Makefile')):
        run('build_unit_tests', base_path, True, enable_cmake)
        run('unit_tests', base_path)
    elif enable_cmake:
        run('unit_tests', base_path)

    run('build_fvm', base_path, True, enable_cmake)
    run('fvm_test', base_path, test_path, True)
    run('fvm_test', base_path, test_path, True, 16)

if __name__ == '__main__':
    main()
