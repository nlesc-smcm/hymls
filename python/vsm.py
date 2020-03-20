import os
import sys
import numpy
import tempfile
import subprocess

def wrtbcsr(beg, jco, co, f):
    n = numpy.int32(len(beg) - 1)

    bc = numpy.int32(n.nbytes)
    print(bc, bc.tobytes())
    os.write(f, bc.tobytes())
    os.write(f, n.tobytes())
    os.write(f, bc.tobytes())

    bc = numpy.int32(beg.nbytes)
    print(bc, bc.tobytes())
    os.write(f, bc.tobytes())
    os.write(f, beg.tobytes())
    os.write(f, bc.tobytes())

    bc = numpy.int32(jco.nbytes)
    print(bc, bc.tobytes())
    os.write(f, bc.tobytes())
    os.write(f, jco.tobytes())
    os.write(f, bc.tobytes())

    bc = numpy.int32(co.nbytes)
    print(bc, bc.tobytes())
    os.write(f, bc.tobytes())
    os.write(f, co.tobytes())
    os.write(f, bc.tobytes())

def main():
    if len(sys.argv) < 2:
        print('Usage: python vsm.py matrix.mtx [nrows] [dof]')

    n = -1
    if len(sys.argv) > 2:
        n = int(sys.argv[2])

    dof = -1
    if len(sys.argv) > 3:
        dof = int(sys.argv[3])

    with open(sys.argv[1], 'r') as f:
        first = True
        idx = 0
        for line in f.readlines():
            spline = line.split(' ')
            if len(spline) != 3:
                continue

            if first:
                if n < 0:
                    n = int(spline[0])
                nnz = int(spline[2])

                rows = list()

                first = False
                continue

            row = int(spline[0]) - 1
            col = int(spline[1])
            val = float(spline[2])
            while len(rows) <= row:
                rows.append(list())
            rows[row].append((col, val))

        n = max(n, len(rows))
        beg = numpy.ndarray(n + 1, numpy.int32)
        jco = numpy.ndarray(nnz, numpy.int32)
        co = numpy.ndarray(nnz, numpy.float)

        idx = 0
        for i in range(n):
            beg[i] = idx + 1
            for j, v in rows[i]:
                jco[idx] = j
                co[idx] = v
                idx += 1
        beg[n] = idx + 1

    f, fname = tempfile.mkstemp()
    wrtbcsr(beg, jco, co, f)
    os.close(f)

    subprocess.call('vsm ' + fname, shell=True)

    os.remove(fname)

if __name__ == '__main__':
    main()
