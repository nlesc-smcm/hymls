import os
import re
import sys
import numpy
import tempfile
import subprocess


def wrtbcsr(beg, jco, co, f):
    n = numpy.int32(len(beg) - 1)

    bc = numpy.int32(n.nbytes)
    os.write(f, bc.tobytes())
    os.write(f, n.tobytes())
    os.write(f, bc.tobytes())

    bc = numpy.int32(beg.nbytes)
    os.write(f, bc.tobytes())
    os.write(f, beg.tobytes())
    os.write(f, bc.tobytes())

    bc = numpy.int32(jco.nbytes)
    os.write(f, bc.tobytes())
    os.write(f, jco.tobytes())
    os.write(f, bc.tobytes())

    bc = numpy.int32(co.nbytes)
    os.write(f, bc.tobytes())
    os.write(f, co.tobytes())
    os.write(f, bc.tobytes())


def main():
    if len(sys.argv) < 2:
        print('Usage: python vsm.py matrix.mtx [nrows] [dof]')

    n = -1
    if len(sys.argv) > 2:
        n = eval(sys.argv[2])

    dof = -1
    if len(sys.argv) > 3:
        dof = int(sys.argv[3])

    index_map = None

    if len(sys.argv) > 4:
        # Parse hid_deb_data
        level = 1
        if len(sys.argv) > 5:
            level = int(sys.argv[5])
        with open(sys.argv[4], 'r') as f:
            data = f.read()
            matches = re.findall('p\{%d\}\{1\}\.groups\{(\d+)\} = (.*?\});' % level, data, flags=re.DOTALL)
            print(matches)
            groups = []
            for i in matches:
                s = i[1].replace('...', '').replace('{', '[').replace('}', ']')
                print(eval(s))
                groups.append(eval(s))

            index = 1
            index_map = [0] * (n+1)
            for i in groups:
                for k in i[0]:
                    k += 1
                    # if k not in index_map:
                    #     index_map.append(k)
                    if index_map[k] == 0:
                        index_map[k] = index
                        index += 1
            for i in groups:
                for j in range(1, len(i)):
                    for k in i[j]:
                        k += 1
                        if index_map[k] == 0:
                            index_map[k] = index
                            index += 1
        print(index_map)
    elif dof > 0:
        # 1 -> 1
        # 4 -> 2
        # 7 -> 3
        index = 1
        index_map = [0] * (n+1)
        for i in range(1, dof+1):
            for j in range(i, n+1, dof):
                index_map[j] = index
                index += 1

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

            row = int(spline[0])
            col = int(spline[1])
            val = float(spline[2])

            # Drop small values in the matrix
            if abs(val) < 1e-15:
                nnz -= 1
                continue

            while len(rows) <= row:
                rows.append(list())
            rows[row].append((col, val))

    # Reindex to group variables together
    if index_map:
        reindexed_rows = list()
        while len(reindexed_rows) <= n:
            reindexed_rows.append(list())

        for i, row in enumerate(rows):
            for col, val in row:
                reindexed_rows[index_map[i]].append((index_map[col], val))
        rows = reindexed_rows

    n = max(n, len(rows)-1)
    beg = numpy.ndarray(n + 1, numpy.int32)
    jco = numpy.ndarray(nnz, numpy.int32)
    co = numpy.ndarray(nnz, float)

    idx = 0
    for i in range(n):
        beg[i] = idx + 1

        if i + 1 >= len(rows):
            continue

        for j, v in rows[i+1]:
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
