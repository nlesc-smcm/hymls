import re
import os
import sys

replace_table = {'Epetra_Comm': ['Epetra_MpiComm', 'Epetra_SerialComm'],
                 'Epetra_RowMatrix': ['Epetra_FECrsMatrix', 'Epetra_CrsMatrix'],
                 'Epetra_Operator': ['Epetra_FECrsMatrix', 'Epetra_CrsMatrix', 'Epetra_RowMatrix'],
                 'Epetra_MultiVector': ['Epetra_Vector']}

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'HYMLS.i.in')) as f:
    orig = f.read()

new = orig

# Add includes for all explicit types
includes = ''
for op in replace_table:
    for rep in replace_table[op]:
        includes += '#include "' + rep + '.h"\n'

new = new.replace('%{\n#include', '%{\n' + includes + '#include')

# Add extra wrappers for all explicit types
for op in replace_table:
    for rep in replace_table[op]:
        matches = re.findall(r'\n\n.*?Teuchos::RCP<' + op + '>.*?\n?.*?\n    {\n.*?\n    }', new, flags=re.MULTILINE)
        for match in matches:
            new_match = match.replace('Teuchos::RCP<' + op + '>', 'Teuchos::RCP<' + rep + '>')
            new = new.replace(match, new_match + match)
            # Remove duplicates if the same method exists in multiple classes
            new = new.replace(new_match + new_match, new_match)

        # Do the same for matches at the start of a class
        matches = re.findall(r'\n{(\n.*?Teuchos::RCP<' + op + '>.*?\n?.*?\n    {\n.*?\n    })', new, flags=re.MULTILINE)
        for match in matches:
            new_match = match.replace('Teuchos::RCP<' + op + '>', 'Teuchos::RCP<' + rep + '>')
            new = new.replace(match, new_match + '\n' + match)
            # Remove duplicates if the same method exists in multiple classes
            new = new.replace(new_match + '\n' + new_match, new_match)

with open(os.path.join(sys.argv[1], 'HYMLS.i'), 'w') as f:
    f.write(new)
