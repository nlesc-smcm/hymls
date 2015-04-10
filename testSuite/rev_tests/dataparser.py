import os
import sys
import matplotlib
import matplotlib.pyplot as pyplot
import copy
import re

class ParserData:
    def __init__(self):
        self.fvm = {'timing': {}, 'failures': 0, 'iterations': 0}
        self.fvm_testing = {'timing': {}, 'failures': 0, 'iterations': 0}

        self.fvm_parallel = {'timing': {}, 'failures': 0, 'iterations': 0}
        self.fvm_testing_parallel = {'timing': {}, 'failures': 0, 'iterations': 0}

        self.integration_tests = {'failures': 0}
        self.integration_tests_testing = {'failures': 0}

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
            if 'failed' in i:
                if time == 1:
                    parsed_data.integration_tests['failures'] += 1
                else:
                    parsed_data.integration_tests_testing['failures'] += 1
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
                    parsed_data.fvm['timing'][name] = d[3].strip()
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

            if i.endswith('out'):
                if'16' not in i:
                    if i.endswith('testing.out'):
                        parsed_data = self.parse_fvm_data(test_data)
                        data.fvm_testing = parsed_data.fvm
                        data.failed = data.failed or parsed_data.failed
                    else:
                        parsed_data = self.parse_fvm_data(test_data)
                        data.fvm = parsed_data.fvm
                        data.failed = data.failed or parsed_data.failed
                else:
                    if i.endswith('testing.out'):
                        parsed_data = self.parse_fvm_data(test_data)
                        data.fvm_testing_parallel = parsed_data.fvm
                        data.failed = data.failed or parsed_data.failed
                    else:
                        parsed_data = self.parse_fvm_data(test_data)
                        data.fvm_parallel = parsed_data.fvm
                        data.failed = data.failed or parsed_data.failed
            elif i == 'integration_tests.txt':
                parsed_data = self.parse_integration_test_data(test_data)
                data.integration_tests = parsed_data.integration_tests
                data.integration_tests_testing = parsed_data.integration_tests_testing
                data.failed = data.failed or parsed_data.failed
            elif i == 'unit_tests.txt':
                parsed_data = self.parse_unit_test_data(test_data)
                data.unit_tests = parsed_data.unit_tests
                data.failed = data.failed or parsed_data.failed

        return data

    def plot_iterations(self):
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
                else:
                    plot.annotate('x', xy=(int(rev), prev_its), xytext=(0,0), textcoords='offset points')
            plot.plot(x, y, linewidth=2)
            yprev = 0
            for (i,j) in zip(x,y):
                if abs(yprev - j) > 0.05 * j:
                    plot.annotate(str(i), xy=(i,j), xytext=(-10, 5), textcoords='offset points')
                yprev = j
            plot.hold(True)
        leg = plot.legend(['FVM', 'FVM (parallel)'], loc=1)
        frame = leg.get_frame()
        frame.set_alpha(0.5)
        plot.set_xlabel(u'rev', size='large')
        plot.set_ylabel(u'time', size='large')
        figure.savefig(os.path.join(self.prefix, 'fvm_iterations.eps'), bbox_inches='tight')
        figure.savefig(os.path.join(self.prefix, 'fvm_iterations.png'), bbox_inches='tight')


    def plot_timing(self, attr='fvm'):
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
                    time = float(getattr(data, attr)['timing'].get(i, '0'))
                    x.append(int(rev))
                    y.append(time)
                    prev_time = time
                else:
                    plot.annotate('x', xy=(int(rev), prev_time), xytext=(0,0), textcoords='offset points')
            plot.plot(x, y, linewidth=2)
            yprev = 0
            for (i,j) in zip(x,y):
                if abs(yprev - j) > 0.05 * j:
                    plot.annotate(str(i), xy=(i,j), xytext=(-10, 5), textcoords='offset points')
                yprev = j
            plot.hold(True)
        leg = plot.legend(data_list, loc=1)
        frame = leg.get_frame()
        frame.set_alpha(0.5)
        plot.set_xlabel(u'rev', size='large')
        plot.set_ylabel(u'time', size='large')
        figure.savefig(os.path.join(self.prefix, attr+'_timing.eps'), bbox_inches='tight')
        figure.savefig(os.path.join(self.prefix, attr+'_timing.png'), bbox_inches='tight')

    def plot_failures(self):
        figure = pyplot.figure()
        plot = figure.add_subplot(111)
        prev_failures = 0
        for attr in ('fvm_testing', 'fvm_testing_parallel', 'unit_tests', 'integration_tests', 'integration_tests_testing'):
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
                else:
                    plot.annotate('x', xy=(int(rev), prev_failures), xytext=(0,0), textcoords='offset points')
            plot.plot(x, y, linewidth=2)
            yprev = 0
            for (i,j) in zip(x,y):
                if abs(yprev - j) > 0.05 * j:
                    plot.annotate(str(i), xy=(i,j), xytext=(-10, 5), textcoords='offset points')
                yprev = j
            plot.hold(True)

        data_list = ['FVM', 'FVM (parallel)', 'unit tests', 'integration tests', 'integration tests (testing)']
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

if __name__ == "__main__":
    main()
