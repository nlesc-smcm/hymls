import sys
import os
import re

def main():
    if len(sys.argv) < 2:
        print 'No filename'
        return

    fname = sys.argv[1]
    if not os.path.exists(fname):
        print fname, 'is not a valid filename'

    f = open(fname, 'r')
    lines = f.readlines()
    f.close()

    if not os.path.exists(fname+'.bak'):
        f = open(fname+'.bak', 'w')
        f.write(''.join(lines))
        f.close()

    text = ''

    in_comment = False

    for line in lines:
        if '/*' in line:
            in_comment = True
        elif '*/' in line:
            in_comment = False
            text += line.rstrip() + '\n'
            continue
        if in_comment:
            text += line.rstrip() + '\n'
            continue

        orig_line = line

        # Put spaces around operators
        ops = ['=', '+', '/', '-', '<', '>']
        for op in ops:
            line = line.replace(op, ' '+op+' ')
            line = line.replace('  '+op, ' '+op)
            line = line.replace(op+'  ', op+' ')

        # Remove spaces around ++ and --
        for op in ['-', '+']:
            line = line.replace(' '+op+' '+op+' ', op+op)

        # Remove spaces between things like // and ==
        for op2 in ['=', '<', '>', '/']:
            for op1 in ['=', '+', '-', '*', '/', '<', '>']:
                line = line.replace(op1+' '+op2, op1+op2)

        # Put spaces after , and ;
        for op in [',', ';']:
            line = line.replace(op, op+' ')
            line = line.replace(op+'  ', op+' ')

        # != is different because we don't want spaces around !
        line = line.replace('! =', '!=')
        line = line.replace('!=', ' != ')
        line = line.replace('!=  ', '!= ')
        line = line.replace('  != ', ' !=')

        # Pointer dereference
        line = line.replace(' - >  ', '->')

        # Spaces after keywords
        for key in ['for', 'if']:
            line = line.replace(key+'(', key+' (')

        # Templates and includes should not have spaces
        line = re.sub(' < ([\w\.<:]+) >', '<\g<1>>', line)

        # Unless nested templates TODO: Fix this for line that have both
        if '> >' in orig_line:
            line = line.replace('>>', '> >')

        # -1 should not have spaces
        line = re.sub('(\W) - ', '\g<1>-', line)
        for op in ops:
            if op != '-':
                line = line.replace(op+'-', op+' -')

        # Comments at the start of a line should stay there
        line = re.sub('^ //', '//', line)

        # Includes should have a space
        line = line.replace('include<', 'include <')

        line = line.rstrip()
        text += line + '\n'

    f = open(fname, 'w')
    f.write(text)
    f.close()

if __name__ == "__main__":
    main()
