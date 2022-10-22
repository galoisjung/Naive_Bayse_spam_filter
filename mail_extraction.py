import random
from imaplib import IMAP4_SSL
import email
import Dao_email
import json

with open('conf.json') as f:
    config = json.load(f)


def findEncodingInfo(txt):
    info = email.header.decode_header(txt)
    s, encoding = info[0]
    return s, encoding


def contents_extract(email):
    result = dict()

    result['From'] = email['From']
    result['To'] = email['To']
    result['Date'] = email['Date']

    if email['Subject'] is not None:
        subject, encode = findEncodingInfo(email['Subject'])
    else:
        subject = " "
        encode = None

    if encode == "unknown-8bit":
        result['Subject'] = str(subject, 'CP949')
    elif encode == "cseuckr":
        result['Subject'] = str(subject, 'euckr')
    elif encode is not None:
        result['Subject'] = str(subject, encode)
    else:
        result['Subject'] = str(subject)

    result['Content'] = dfs(email)

    return result


def dfs(email_cont, stack=[]):
    result = str()
    stack.append(email_cont)
    while len(stack) > 0:
        print(len(stack))
        email_cont = stack.pop()
        if email_cont.is_multipart():
            stack.extend(email_cont.get_payload())
            email_cont = stack.pop()
            result += dfs(email_cont, stack)
        else:
            byte = email_cont.get_payload(decode=True)
            encode = email_cont.get_content_charset()

            if encode == "unknown-8bit":
                result = str(byte, 'CP949')
            elif encode == "cseuckr":
                result['Subject'] = str(byte, 'euckr')
            elif encode is not None:
                result = "\n" + str(byte, encode)
            else:
                result = "\n"

    return result


def spam_extraction():
    mail = IMAP4_SSL("imap.gmail.com")
    mail.login(config['GMAIL_ID'], config['GMAIL_PASSWORD'])
    mail.select("[Gmail]/&yRHGlA-")

    resp, data = mail.uid('search', None, 'All')

    all_email = data[0].split()

    for i in all_email:
        print(i)
        result, maildata = mail.uid('fetch', i, '(RFC822)')
        raw_email = maildata[0][1]
        email_message = email.message_from_bytes(raw_email)

        email_obj = contents_extract(email_message)
        Dao_email.spam_add(email_obj)


def ham_extraction():
    mail = IMAP4_SSL("imap.gmail.com")
    mail.login(config['GMAIL_ID'], config['GMAIL_PASSWORD'])
    mail.select("INBOX")

    resp, data = mail.uid('search', None, 'All')

    all_email = data[0].split()

    for i in all_email:
        print(i)
        result, maildata = mail.uid('fetch', i, '(RFC822)')
        raw_email = maildata[0][1]
        email_message = email.message_from_bytes(raw_email)

        email_obj = contents_extract(email_message)
        Dao_email.ham_add(email_obj)


def making_doclist(per):
    hamzip = Dao_email.ham_get()
    spamzip = Dao_email.spam_get()

    hamlist = list(set([i for i in hamzip]))
    spamlist = list(set([i for i in spamzip]))

    resultham = []
    resultspam = []

    resultham.extend(hamlist)
    resultspam.extend(spamlist)

    train_set = random.sample(hamlist, int(len(hamlist) * per))
    test_set = list(set(hamlist).difference(train_set))

    train_set.extend(random.sample(spamlist, int(len(spamlist) * per)))
    test_set.extend(set(spamlist).difference(train_set))

    return train_set, test_set
