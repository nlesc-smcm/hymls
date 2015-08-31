import sys
import os
import re

class StringReplacer(object):
    Normal = 0
    String = 1
    Comment = 2
    MultilineComment = 3

    def __init__(self, text, type):
        self.text = text
        self.type = type

    def replace(self, search, replace):
        if self.type == self.Normal:
            self.text = self.text.replace(search, replace)

    def regex_replace(self, search, replace):
        if self.type == self.Normal:
            self.text = re.sub(search, replace, self.text)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

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
    line_type = StringReplacer.Normal

    for line_num, line in enumerate(lines):
        orig_line = line
        line_parts = []
        line_part = ''
        line_type = line_type if line_type == StringReplacer.MultilineComment else StringReplacer.Normal
        for pos, char in enumerate(orig_line):
            line_part += char
            if char == '"':
                line_parts.append(StringReplacer(line_part, line_type))
                line_type = StringReplacer.Normal if line_type == StringReplacer.String else StringReplacer.String
                line_part = ''
                continue

            if line_part.endswith('//'):
                line_parts.append(StringReplacer(line_part[:-2], line_type))
                line_part = orig_line[pos-1:]
                break

            if line_part.endswith('/*'):
                line_parts.append(StringReplacer(line_part[:-2], line_type))
                line_type = StringReplacer.MultilineComment
                line_part = '/*'
                continue

            if line_part.endswith('*/'):
                line_parts.append(StringReplacer(line_part, line_type))
                line_type = StringReplacer.Normal
                line_part = ''
                continue

        line_parts.append(StringReplacer(line_part, line_type))

        line = ''
        for line_part in line_parts:
            if line_part.type != StringReplacer.Normal:
                line += str(line_part)
                continue

            # Put spaces around operators
            ops = ['=', '+', '/', '-', '<', '>']
            for op in ops:
                line_part.replace(op, ' '+op+' ')
                line_part.replace('  '+op, ' '+op)
                line_part.replace(op+'  ', op+' ')

            # Remove spaces around ++ and --
            for op in ['-', '+']:
                line_part.replace(' '+op+' '+op+' ', op+op)

            # Remove spaces between things like // and ==
            for op2 in ['=', '<', '>', '/']:
                for op1 in ['=', '+', '-', '*', '/', '<', '>']:
                    line_part.replace(op1+' '+op2, op1+op2)

            # != is different because we don't want spaces around !
            line_part.replace('! =', '!=')
            line_part.replace('!=', ' != ')
            line_part.replace('!=  ', '!= ')
            line_part.replace('  !=', ' !=')

            # Pointer dereference
            line_part.replace(' - >  ', '->')

            # Remove spaces around operators
            for op in ['->']:
                line_part.replace(' '+op, op)
                line_part.replace(op+' ', op)

            # Spaces after keywords
            for key in ['for', 'if']:
                line_part.replace(key+'(', key+' (')

            # Templates and includes should not have spaces
            line_part.regex_replace(' < ([\w\.<: ]+) >', '<\g<1>>')

            # Unless nested templates TODO: Fix this for line_part that have both
            if '> >' in orig_line:
                line_part.replace('>>', '> >')

            # Template members
            line_part.replace('> ::', '>::')

            # -1 should not have spaces
            line_part.regex_replace('([^\w\]\)]) - ', '\g<1>-')
            for op in ops:
                if op != '-':
                    line_part.replace(op+'-', op+' -')

            # Put spaces after , and ;
            for op in [',', ';']:
                line_part.replace(op, op+' ')
                line_part.replace(op+'  ', op+' ')

            # Comments at the start of a line_part should stay there
            line_part.regex_replace('^ //', '//')

            # Includes should have a space
            line_part.replace('include<', 'include <')

            line += str(line_part)

        line = line.rstrip()
        text += line + '\n'

    f = open(fname, 'w')
    f.write(text)
    f.close()

if __name__ == "__main__":
    main()
