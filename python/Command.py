"""This file contains classes that implement a thread that runs a command.
The process can then be killed after a certain time.
"""

import subprocess
import threading
import datetime
import os
from collections import OrderedDict

class Command(object):
    def __init__(self, cmd, env=None):
        self.cmd = cmd
        self.process = None
        self.out = ''
        self.err = ''
        self.env = env
        if self.env is not None:
            self.env.update(os.environ)

    def run(self, timeout=600):
        def target():
            self.out += 'Thread started\n'
            self.out += 'Running ' + self.cmd + '\n'
            self.process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable="/bin/bash", env=self.env)
            (out, err) = self.process.communicate()
            if out:
                self.out += out
            if err:
                self.err += err
            self.out += 'Thread finished at ' + str(datetime.datetime.now()) + '\n'

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        killed = False
        if thread.is_alive():
            self.kill()
            thread.join()
            killed = True

        if self.process is None:
            return (-1, killed)

        self.out += 'Returncode is ' + str(self.process.returncode) + '\n'
        return (self.process.returncode, killed)

    def kill(self):
        self.out += 'Terminating process\n'
        subprocess.call('killall -9 '+self.cmd.partition(' ')[0], shell=True)

class ParallelCommand(Command):
    def __init__(self, cmd, env=None, procs=1, nodes=1):
        Command.__init__(self, cmd, env)

        self.procs = procs
        self.nodes = nodes
        np = procs // nodes

        self.mpis = OrderedDict([('mpiexec', 'mpiexec -n %d -npernode %d' % (self.procs, np)), ('mpirun', 'mpirun -n %d -npernode %d' % (self.procs, np)), ('srun', 'srun --nodes=%d --ntasks=%d --ntasks-per-node=%d' % (self.nodes, self.procs, np))])
        self.orig_cmd = cmd
        self.mpi = None

        for mpi in self.mpis.iterkeys():
            p = subprocess.Popen(mpi+' --help', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable="/bin/bash", env=self.env)
            p.communicate()
            if p.returncode == 0:
                self.cmd = self.mpis[mpi] + ' ' + cmd
                self.mpi = mpi
                break

    def kill(self):
        super(ParallelCommand, self).kill()
        if self.mpi:
            subprocess.call('killall -9 '+self.mpi, shell=True)
        if self.orig_cmd:
            subprocess.call('killall -9 '+self.orig_cmd.partition(' ')[0], shell=True)
