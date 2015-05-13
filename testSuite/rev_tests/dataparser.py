import os
import sys
import matplotlib
import matplotlib.pyplot as pyplot
import copy
import re
import subprocess

def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return 0

def get_max(arr1, arr2=None):
    arrmax = 0
    if arr1:
        arrmax = max(arrmax, max(arr1))
    if arr2:
        arrmax = max(arrmax, max(arr2))
    return arrmax

class ParserData:
    def __init__(self):
        self.fvm = {'timing': {}, 'failures': 0, 'iterations': 0}
        self.fvm_testing = {'timing': {}, 'failures': 0, 'iterations': 0}

        self.fvm_parallel = {'timing': {}, 'failures': 0, 'iterations': 0}
        self.fvm_testing_parallel = {'timing': {}, 'failures': 0, 'iterations': 0}

        self.integration_tests = {'failures': 0}
        self.integration_tests_parallel = {'failures': 0}

        self.unit_tests = {'failures': 0}

        self.failed = False

class Parser:
    def __init__(self, prefix=None):
        self.prefix = prefix
        self.data = {}

    def parse(self):
        for i in os.listdir(self.prefix):
            path = os.path.join(self.prefix, i)
            if os.path.isdir(path) and len(i) < 5 and i.isdigit():
                print i
                self.data[i] = self.parse_dir(path)
                #~ print self.data[i].fvm

    def parse_integration_test_data(self, data):
        parsed_data = ParserData()
        time = 0
        for i in data:
            if 'Starting integration tests' in i:
                time += 1
            if re.match('^Test.+failed.$', i):
                if time == 1:
                    parsed_data.integration_tests['failures'] += 1
                else:
                    parsed_data.integration_tests_parallel['failures'] += 1
        return parsed_data

    def parse_unit_test_data(self, data):
        parsed_data = ParserData()
        for i in data:
            if 'Summary: ' in i:
                parsed_data.unit_tests['failures'] = int(i.split(' ')[-1])
        return parsed_data

    def parse_fvm_data(self, data):
        timing_started = False
        parsed_data = ParserData()
        for i in data:
            if 'TIMING RESULTS' in i:
                timing_started = True
            if timing_started:
                d = i.split('\t')
                if len(d) == 4:
                    name = re.sub('_L(\d)', ' (level \g<1>)', d[0])
                    parsed_data.fvm['timing'][name] = parse_float(d[3].strip())
            if 'FAILED!' in i:
                parsed_data.fvm['failures'] += 1
            if i.startswith('+ Number of iterations:'):
                parsed_data.fvm['iterations'] = int(i[23:])

        # If we don't have timing data something went wrong
        if not parsed_data.fvm['timing']:
            parsed_data.failed = True

        return parsed_data

    def parse_dir(self, path):
        data = ParserData()
        for i in os.listdir(path):
            test_path = os.path.join(path, i)
            test_file = open(test_path, 'r')
            test_data = test_file.readlines()
            test_file.close()

            if not i.endswith('out') and not i == 'integration_tests.txt' \
               and not i == 'unit_tests.txt':
                continue

            if i.endswith('out'):
                if'16' not in i:
                    if i.endswith('testing.out'):
                        parsed_data = self.parse_fvm_data(test_data)
                        data.fvm_testing = parsed_data.fvm
                    else:
                        parsed_data = self.parse_fvm_data(test_data)
                        data.fvm = parsed_data.fvm
                else:
                    if i.endswith('testing.out'):
                        parsed_data = self.parse_fvm_data(test_data)
                        data.fvm_testing_parallel = parsed_data.fvm
                    else:
                        parsed_data = self.parse_fvm_data(test_data)
                        data.fvm_parallel = parsed_data.fvm
            elif i == 'integration_tests.txt':
                parsed_data = self.parse_integration_test_data(test_data)
                data.integration_tests = parsed_data.integration_tests
                data.integration_tests_parallel = parsed_data.integration_tests_parallel
            elif i == 'unit_tests.txt':
                parsed_data = self.parse_unit_test_data(test_data)
                data.unit_tests = parsed_data.unit_tests

            data.failed = data.failed or parsed_data.failed

            # Check for crashes
            rets = re.findall('Returncode is (\d+)', ''.join(test_data))
            if not rets:
                data.failed = True
            for ret in rets:
                if ret != '0':
                    data.failed = True

        return data

    def plot_iterations(self):
        last = 0
        figure = pyplot.figure()
        plot = figure.add_subplot(111)
        prev_its = 0
        for attr in ('fvm', 'fvm_parallel'):
            x = []
            y = []
            for rev in sorted(self.data.keys(), key=int):
                data = self.data[rev]
                if not data.failed:
                    its = int(getattr(data, attr)['iterations'])
                    x.append(int(rev))
                    y.append(its)
                    prev_its = its
                    last = rev
                else:
                    plot.annotate('x', xy=(int(rev), prev_its), xytext=(0,0), textcoords='offset points')
            plot.plot(x, y, linewidth=2)
            yprev = 0
            for (i,j) in zip(x,y):
                if abs(yprev - j) > 0.05 * j:
                    plot.annotate(str(i), xy=(i,j), xytext=(-10, 5), textcoords='offset points')
                yprev = j
            plot.hold(True)

        plot.set_title('Iterations up to revision '+str(last))
        leg = plot.legend(['FVM', 'FVM (parallel)'], loc=1)
        frame = leg.get_frame()
        frame.set_alpha(0.5)
        plot.set_xlabel(u'rev', size='large')
        plot.set_ylabel(u'iterations', size='large')
        figure.savefig(os.path.join(self.prefix, 'fvm_iterations.eps'), bbox_inches='tight')
        figure.savefig(os.path.join(self.prefix, 'fvm_iterations.png'), bbox_inches='tight')


    def plot_timing(self, attr='fvm'):
        last = 0
        figure = pyplot.figure()
        plot = figure.add_subplot(111)
        data_list = ['main: continuation run', 'HYMLS::Solver: ApplyInverse', 'Preconditioner (level 1): Compute']
        prev_time = 0
        for i in data_list:
            x = []
            y = []
            for rev in sorted(self.data.keys(), key=int):
                data = self.data[rev]
                #~ print 'data', data.fvm
                #~ print rev
                if not data.failed:
                    time = getattr(data, attr)['timing'].get(i, 0)
                    x.append(int(rev))
                    y.append(time)
                    prev_time = time
                    last = rev
                else:
                    plot.annotate('x', xy=(int(rev), prev_time), xytext=(0,0), textcoords='offset points')
            plot.plot(x, y, linewidth=2)
            yprev = 0
            for (i,j) in zip(x,y):
                if abs(yprev - j) > 0.05 * j:
                    plot.annotate(str(i), xy=(i,j), xytext=(-10, 5), textcoords='offset points')
                yprev = j
            plot.hold(True)

        plot.set_title('Performance up to revision '+str(last)+(' (parallel)' if attr=='fvm_parallel' else ''))
        leg = plot.legend(data_list, loc=1)
        frame = leg.get_frame()
        frame.set_alpha(0.5)
        plot.set_xlabel(u'rev', size='large')
        plot.set_ylabel(u'time', size='large')
        figure.savefig(os.path.join(self.prefix, attr+'_timing.eps'), bbox_inches='tight')
        figure.savefig(os.path.join(self.prefix, attr+'_timing.png'), bbox_inches='tight')

    def plot_failures(self):
        last = 0
        figure = pyplot.figure()
        plot = figure.add_subplot(111)
        prev_failures = 0
        for attr in ('fvm_testing', 'fvm_testing_parallel', 'unit_tests', 'integration_tests', 'integration_tests_parallel'):
            x = []
            y = []
            for rev in sorted(self.data.keys(), key=int):
                data = self.data[rev]
                #~ print rev
                if not data.failed:
                    failures = float(getattr(data, attr)['failures'])
                    x.append(int(rev))
                    y.append(failures)
                    prev_failures = failures
                    last = rev
                else:
                    plot.annotate('x', xy=(int(rev), prev_failures), xytext=(0,0), textcoords='offset points')
            plot.plot(x, y, linewidth=2)
            yprev = 0
            for (i,j) in zip(x,y):
                if abs(yprev - j) > 0.05 * j:
                    plot.annotate(str(i), xy=(i,j), xytext=(-10, 5), textcoords='offset points')
                yprev = j
            plot.hold(True)

        plot.set_title('Failures up to revision '+str(last))
        data_list = ['FVM', 'FVM (parallel)', 'unit tests', 'integration tests', 'integration tests (parallel)']
        leg = plot.legend(data_list, loc=1)
        frame = leg.get_frame()
        frame.set_alpha(0.5)
        plot.set_xlabel(u'rev', size='large')
        plot.set_ylabel(u'failures', size='large')
        figure.savefig(os.path.join(self.prefix, 'test_failures.eps'), bbox_inches='tight')
        figure.savefig(os.path.join(self.prefix, 'test_failures.png'), bbox_inches='tight')

def main():
    parser = Parser('./')
    parser.parse()
    parser.plot_timing('fvm')
    parser.plot_timing('fvm_parallel')
    parser.plot_iterations()
    parser.plot_failures()

def find_previous_commit(commit):
    if commit.isdigit() and len(commit) < 5:
        p = subprocess.Popen('git svn find-rev r'+commit, stdout=subprocess.PIPE, shell=True)
        (out, err) = p.communicate()
        commit = out.strip()

    p = subprocess.Popen('git log --pretty=oneline --format="%H" '+commit+'^ --', stdout=subprocess.PIPE, shell=True)
    (out, err) = p.communicate()
    commits = out.strip().split('\n')
    d = os.listdir('./')
    for i in commits:
        if i in d:
            return i
        p = subprocess.Popen('git svn find-rev '+i, stdout=subprocess.PIPE, shell=True)
        (out, err) = p.communicate()
        rev = out.strip()
        if rev in d:
            return rev
    return ''

def compare(first, second=None):
    if second is None:
        second = first
        first = find_previous_commit(second)

    if not (os.path.isdir(first) and os.path.isdir(second)):
        message = 'No valid directories/revisions supplied'
        print message
        return message

    parser = Parser('./')
    data1 = parser.parse_dir(first)
    data2 = parser.parse_dir(second)

    errors = ''
    warnings = ''
    report = ''

    # Bad hack for iterating over members of ParserData
    for name in data1.__dict__.iterkeys():
        subdata1 = getattr(data1, name)
        subdata2 = getattr(data2, name)
        if name == 'failed':
            if subdata2:
                errors += 'Error: Tests failed\n'
            continue

        # Search for values that changed (too much) compared to the previous commit
        for item, value in subdata1.iteritems():
            value2 = subdata2[item]
            if isinstance(value, int):
                if value != value2:
                    warnings += 'Warning: {:s} in {:s} went from {:d} to {:d}\n'.format(
                        item, name, value, value2)
            if item == 'timing':
                maxtime = get_max(value.values(), value2.values())
                for tname, t in value.iteritems():
                    t2 = value2.get(tname, 0)
                    # if t2 == 0 or t2 differs from t for more than 20%, assume failure
                    if t > maxtime * 0.1 and (abs(t2) < 1e-8 or abs(t2 / t - 1) > 0.2):
                        # long name
                        short_tname = tname.replace('AssembleTransformAndDrop (first call)', 'AT&D')
                        report += 'Warning: {:<50s} in {:s}: {:6.2f}s, was {:6.2f}s\n'.format(
                            short_tname, name, t, subdata1.get(item, {}).get(tname, 0))

        # Make a report containing all relevant values
        report += '\n' + name + '\n'
        for item, value in subdata2.iteritems():
            if isinstance(value, int):
                report += '{:s}: {:d}\n'.format(item, value)
            if item == 'timing':
                maxtime = get_max(value.values())
                for tname, t in value.iteritems():
                    if t > maxtime * 0.1:
                        # long name
                        short_tname = tname.replace('AssembleTransformAndDrop (first call)', 'AT&D')
                        report += '{:<50s}: {:6.2f}s, was {:6.2f}s\n'.format(short_tname, t, subdata1.get(item, {}).get(tname, 0))

    message = 'Tests completed succesfully'
    if errors or warnings:
        message = errors + '\n' + warnings

    message += '\n\n------------------------\nFull report\n------------------------\n' + report

    print message
    return message

if __name__ == "__main__":
    if len(sys.argv) == 3:
        compare(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        compare(sys.argv[1])
    else:
        main()
