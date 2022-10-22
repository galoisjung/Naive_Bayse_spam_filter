import json
from imaplib import IMAP4_SSL
import email
import Dao_email

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


def spam_extraction(connection):
    mail = IMAP4_SSL("imap.naver.com")
    mail.login(config['NAVER_ID'], config['NAVER_PASSWORD'])
    mail.select("Junk")

    resp, data = mail.uid('search', None, 'All')

    all_email = data[0].split()

    for i in all_email:
        print(i)
        result, maildata = mail.uid('fetch', i, '(RFC822)')
        raw_email = maildata[0][1]
        email_message = email.message_from_bytes(raw_email)

        email_obj = contents_extract(email_message)
        Dao_email.add(email_obj, connection, True)


def ham_extraction(connection):
    mail = IMAP4_SSL("imap.naver.com", port=993)
    mail.login(config['NAVER_ID'], config['NAVER_PASSWORD'])
    mail.select("INBOX")

    resp, data = mail.uid('search', None, 'All')

    all_email = data[0].split()

    for i in all_email:
        print(i)
        result, maildata = mail.uid('fetch', i, '(RFC822)')
        raw_email = maildata[0][1]
        email_message = email.message_from_bytes(raw_email)

        email_obj = contents_extract(email_message)
        Dao_email.add(email_obj, connection, False)


def making_doclist(per, connection):
    hamzip = Dao_email.ham_get(connection)
    spamzip = Dao_email.spam_get(connection)

    hamlist = [i for i in hamzip]
    spamlist = [i for i in spamzip]

    resultham = []
    resultspam = []

    resultham.extend(hamlist)
    resultspam.extend(spamlist)

    train_set = []
    test_set = []

    train_set.extend(hamlist[0: int(len(hamlist) * per)])
    test_set.extend(hamlist[int(len(hamlist) * per):])

    train_set.extend(spamlist[0: int(len(spamlist) * per)])
    test_set.extend(spamlist[int(len(spamlist) * per):])

    return train_set, test_set
