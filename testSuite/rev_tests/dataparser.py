import os
import sys
import matplotlib
import matplotlib.pyplot as pyplot
import copy

class ParserData:
    def __init__(self):
        self.fvm = {'timing': {}, 'failures': 0}
        self.fvm_testing = {'timing': {}, 'failures': 0}

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
                    parsed_data['timing'][d[0]] = d[3].strip()
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
                if i.endswith('testing.out'):
                    data.fvm_testing = self.parse_fvm_data(fvm_data)
                else:
                    data.fvm = self.parse_fvm_data(fvm_data)
        return data

    def plot_timing(self):
        figure = pyplot.figure()
        plot = figure.add_subplot(111)
        data_list = ['main: continuation run', 'HYMLS::Solver: ApplyInverse', 'Preconditioner_L1: Compute']
        for i in data_list:
            x = []
            y = []
            for rev in sorted(self.data.keys(), key=int):
                data = self.data[rev]
                print 'data', data.fvm
                print rev
                x.append(int(rev))
                y.append(float(data.fvm['timing'].get(i, '0')))
            plot.plot(x, y, linewidth=2)
            plot.hold(True)
        leg = plot.legend(data_list, loc=1)
        frame = leg.get_frame()
        frame.set_alpha(0.5)
        plot.set_xlabel(u'rev', size='large')
        plot.set_ylabel(u'time', size='large')
        figure.savefig(os.path.join(self.prefix, 'fvm_timing.eps'), bbox_inches='tight')
        figure.savefig(os.path.join(self.prefix, 'fvm_timing.png'), bbox_inches='tight')


    def plot_failures(self):
        figure = pyplot.figure()
        plot = figure.add_subplot(111)
        x = []
        y = []
        for rev in sorted(self.data.keys(), key=int):
            data = self.data[rev]
            print rev
            x.append(int(rev))
            y.append(float(data.fvm_testing['failures']))
        plot.plot(x, y, linewidth=2)
        plot.hold(True)
        plot.set_xlabel(u'rev', size='large')
        plot.set_ylabel(u'time', size='large')
        figure.savefig(os.path.join(self.prefix, 'fvm_failures.eps'), bbox_inches='tight')
        figure.savefig(os.path.join(self.prefix, 'fvm_failures.png'), bbox_inches='tight')

def main():
    parser = Parser('./')
    parser.parse()
    parser.plot_timing()
    parser.plot_failures()

if __name__ == "__main__":
    main()
