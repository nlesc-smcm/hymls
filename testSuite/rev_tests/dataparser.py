import os
import sys
import matplotlib
import matplotlib.pyplot as pyplot
import copy
import re

class ParserData:
    def __init__(self):
        self.fvm = {'timing': {}, 'failures': 0}
        self.fvm_testing = {'timing': {}, 'failures': 0}

        self.fvm_parallel = {'timing': {}, 'failures': 0}
        self.fvm_testing_parallel = {'timing': {}, 'failures': 0}

        self.integration_tests = {'failures': 0}
        self.integration_tests_testing = {'failures': 0}

class Parser:
    def __init__(self, prefix=None):
        self.prefix = prefix
        self.data = {}

    def parse(self):
        for i in os.listdir(self.prefix):
            print i
            path = os.path.join(self.prefix, i)
            if os.path.isdir(path) and len(i) < 5:
                self.data[i] = self.parse_dir(path)
                print self.data[i].fvm

    def parse_integration_test_data(self, data):
        parsed_data = ParserData()
        first = True
        for i in data:
            if 'TESTS OUT OF' in i:
                print i
                if first:
                    parsed_data.integration_tests['failures'] = int(i[9:11])
                    first = False
                else:
                    parsed_data.integration_tests_testing['failures'] = int(i[9:11])
        return parsed_data

    def parse_fvm_data(self, data):
        timing_started = False
        parsed_data = {'timing': {}, 'failures': 0}
        for i in data:
            if 'TIMING RESULTS' in i:
                timing_started = True
            if timing_started and 'MEMORY USAGE' in i:
                started = False
            if timing_started:
                d = i.split('\t')
                if len(d) == 4:
                    name = re.sub('_L(\d)', ' (level \g<1>)', d[0])
                    parsed_data['timing'][name] = d[3].strip()
            if 'FAILED!' in i:
                parsed_data['failures'] += 1
        return parsed_data

    def parse_dir(self, path):
        data = ParserData()
        for i in os.listdir(path):
            if i.endswith('out'):
                print i
                fvm_path = os.path.join(path, i)
                fvm_file = open(fvm_path, 'r')
                fvm_data = fvm_file.readlines()
                fvm_file.close()
                if'16' not in i:
                    if i.endswith('testing.out'):
                        data.fvm_testing = self.parse_fvm_data(fvm_data)
                    else:
                        data.fvm = self.parse_fvm_data(fvm_data)
                else:
                    if i.endswith('testing.out'):
                        data.fvm_testing_parallel = self.parse_fvm_data(fvm_data)
                    else:
                        data.fvm_parallel = self.parse_fvm_data(fvm_data)
            elif i == 'integration_tests.txt':
                test_path = os.path.join(path, i)
                test_file = open(test_path, 'r')
                test_data = test_file.readlines()
                test_file.close()
                parsed_data = self.parse_integration_test_data(test_data)
                data.integration_tests = parsed_data.integration_tests
                data.integration_tests_testing = parsed_data.integration_tests_testing

        return data

    def plot_timing(self, attr='fvm'):
        figure = pyplot.figure()
        plot = figure.add_subplot(111)
        data_list = ['main: continuation run', 'HYMLS::Solver: ApplyInverse', 'Preconditioner (level 1): Compute']
        for i in data_list:
            x = []
            y = []
            for rev in sorted(self.data.keys(), key=int):
                data = self.data[rev]
                print 'data', data.fvm
                print rev
                x.append(int(rev))
                y.append(float(getattr(data, attr)['timing'].get(i, '0')))
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

        for attr in ('fvm_testing', 'fvm_testing_parallel', 'integration_tests', 'integration_tests_testing'):
            x = []
            y = []
            for rev in sorted(self.data.keys(), key=int):
                data = self.data[rev]
                print rev
                x.append(int(rev))
                y.append(float(getattr(data, attr)['failures']))
            plot.plot(x, y, linewidth=2)
            yprev = 0
            for (i,j) in zip(x,y):
                if abs(yprev - j) > 0.05 * j:
                    plot.annotate(str(i), xy=(i,j), xytext=(-10, 5), textcoords='offset points')
                yprev = j
            plot.hold(True)

        data_list = ['FVM', 'FVM (parallel)', 'integration tests', 'integration tests (testing)']
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
    parser.plot_failures()

if __name__ == "__main__":
    main()
