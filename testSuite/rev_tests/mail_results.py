from mailer import Mailer
from mailer import Message
import sys
import dataparser

def main():
    if len(sys.argv) < 2:
        return

    rev = sys.argv[1]

    message = Message(From="s.baars@rug.nl",
                      To="s.baars@rug.nl",
                      charset="utf-8")
    message.Subject = "HYMLS test results for revision " + rev
    message.Body = dataparser.compare(rev)

    sender = Mailer('smtp.rug.nl')
    sender.send(message)

if __name__ == "__main__":
    main()
